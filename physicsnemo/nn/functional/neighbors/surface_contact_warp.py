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

from bisect import bisect_left, bisect_right

import torch
import warp as wp


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
    queries intersecting a page are revisited. The O(N) prefix offsets are copied
    to CPU once, not a potentially O(NF) list of candidate pairs.
    """
    wp.init()
    device = wp.device_from_torch(lower.device)
    stream = (
        wp.stream_from_torch(torch.cuda.current_stream(lower.device))
        if lower.is_cuda
        else None
    )
    with wp.ScopedDevice(device), wp.ScopedStream(stream):
        wl = wp.from_torch(lower.contiguous(), dtype=wp.vec3)
        wu = wp.from_torch(upper.contiguous(), dtype=wp.vec3)
        tree = wp.Bvh(wl, wu)
        ql = wp.from_torch(query_lower.contiguous(), dtype=wp.vec3)
        qu = wp.from_torch(query_upper.contiguous(), dtype=wp.vec3)
        wf, wb = wp.from_torch(faces.contiguous()), wp.from_torch(batch.contiguous())
        counts = torch.zeros(len(query_lower), dtype=torch.long, device=lower.device)
        args = [tree.id, ql, qu, wf, wb]
        wp.launch(
            _count,
            dim=len(counts),
            inputs=args + [wp.from_torch(counts)],
            device=device,
            stream=stream,
        )
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        host_offsets = offsets.cpu().tolist()
        total = host_offsets[-1]
        if max_pairs is not None and total > max_pairs:
            raise RuntimeError(
                f"surface candidate budget exceeded ({total}>{max_pairs}); no pairs truncated"
            )
    for start in range(0, total, pair_chunk_size):
        stop = min(start + pair_chunk_size, total)
        first = bisect_right(host_offsets, start) - 1
        last = bisect_left(host_offsets, stop)
        # Never keep a Warp device/stream context active while yielding to the
        # caller (which may launch other work or abandon the iterator).
        with wp.ScopedDevice(device), wp.ScopedStream(stream):
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


def bvh_pairs(lower, upper, query_lower, query_upper, faces, batch, max_pairs):
    """Compatibility collector retaining the original fail-closed total budget."""
    chunks = list(
        bvh_pair_chunks(
            lower, upper, query_lower, query_upper, faces, batch, max_pairs, max_pairs
        )
    )
    return torch.cat(chunks, 1) if chunks else faces.new_empty((2, 0))
