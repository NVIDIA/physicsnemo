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
Warp kernels for the Morton-grid radius search backend.

This module is pure Warp code (no PyTorch). It implements the building blocks
for the Morton-sorted spatial grid + custom open-addressed hash design:

* ``@wp.func`` helpers: variable-width 3D Morton encode, batch/cell key packing,
  a 64-bit mix hash, and the open-addressed ``lookup_cell`` probe.
* PC-index build kernels: key generation, sorted-array gather, and hash build.
* Query-side gather and the tiled-task generation kernels.
* The three search kernels: scalar-hash (A), tiled-FMA (B1), tiled-GEMM (B2).

The host glue that drives these kernels lives in ``_morton_impl.py``. Cell
lookup uses a custom open-addressed hash table (built with ``wp.atomic_cas``);
the unique cell keys are also stored sorted, so a binary search over
``cell_key`` is a drop-in alternative for :func:`lookup_cell` if a build lacks
``wp.atomic_cas``.
"""

import warp as wp

# Tile sizes for the tiled (B1/B2) kernels. These are module-level Python ints
# so they are captured as compile-time constants in the Warp kernels (tile
# shapes must be known at code-generation time) while remaining usable directly
# in the host launch code. ``TILE_P`` doubles as ``block_dim`` for the GEMM
# tiled launch (one lane per candidate point) and as the point-chunk width for
# the FMA launch; ``TILE_Q`` is the query-tile width.
#
# ``TILE_P`` must equal Warp's tile block width (default 256) so that
# ``wp.untile`` in the GEMM kernel sees a last tile dimension matching the block
# width when the module is code-generated. Using 256 keeps module compilation
# (default block dim) and the ``launch_tiled(block_dim=TILE_P)`` launch in sync.
TILE_Q = 8
TILE_P = 256


# ---------------------------------------------------------------------------
# Device helper functions
# ---------------------------------------------------------------------------


@wp.func
def morton_encode(x: wp.int32, y: wp.int32, z: wp.int32, bits: wp.int32) -> wp.int64:
    """
    Interleave the low ``bits`` of three non-negative integer coordinates into a
    3D Morton (Z-order) code.

    A loop-based interleave is used (instead of fixed "magic-bits" masks) so the
    same function supports any ``bits`` per axis up to 21, letting the host pick
    the minimum width that covers the data extent (and leaves room for the batch
    id in the high bits of an ``int64`` key).

    Args:
        x: Non-negative cell coordinate along x.
        y: Non-negative cell coordinate along y.
        z: Non-negative cell coordinate along z.
        bits: Number of low bits to interleave per axis.

    Returns:
        The interleaved Morton code as a non-negative ``int64``.
    """
    m = wp.int64(0)
    for i in range(bits):
        bx = wp.int64((x >> i) & 1)
        by = wp.int64((y >> i) & 1)
        bz = wp.int64((z >> i) & 1)
        shift = wp.int64(3 * i)
        m = m | (bx << shift) | (by << (shift + wp.int64(1))) | (bz << (shift + wp.int64(2)))
    return m


@wp.func
def pack_key(
    b: wp.int32,
    cx: wp.int32,
    cy: wp.int32,
    cz: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
) -> wp.int64:
    """
    Pack a batch id and integer cell coordinates into a single ``int64`` key.

    The Morton code occupies the low ``morton_bits`` bits and the batch id is
    shifted above it. The host guarantees ``morton_bits + ceil(log2(B))`` stays
    within the positive ``int64`` range.

    Args:
        b: Batch index.
        cx: Non-negative cell coordinate along x.
        cy: Non-negative cell coordinate along y.
        cz: Non-negative cell coordinate along z.
        bits: Number of bits per axis used by the Morton encode.
        morton_bits: Total Morton-code bit width (``3 * bits``).

    Returns:
        The packed ``int64`` cell key.
    """
    m = morton_encode(cx, cy, cz, bits)
    return (wp.int64(b) << wp.int64(morton_bits)) | m


@wp.func
def hash64(x: wp.uint64) -> wp.uint64:
    """
    64-bit integer finalizer mix (the MurmurHash3 ``fmix64`` constants).

    Args:
        x: Input key bits.

    Returns:
        A well-mixed 64-bit hash of ``x``.
    """
    x = x ^ (x >> wp.uint64(33))
    x = x * wp.uint64(0xFF51AFD7ED558CCD)
    x = x ^ (x >> wp.uint64(33))
    x = x * wp.uint64(0xC4CEB9FE1A85EC53)
    x = x ^ (x >> wp.uint64(33))
    return x


@wp.func
def lookup_cell(
    key: wp.int64,
    hash_slot: wp.array(dtype=wp.int32),
    cell_key: wp.array(dtype=wp.int64),
    hash_cap: wp.int32,
) -> wp.int32:
    """
    Look up the occupied-cell slot for a packed cell ``key``.

    Linear-probes the open-addressed hash table. ``hash_slot`` stores occupied
    cell indices (or ``-1`` for empty), and the full key is verified against
    ``cell_key``. No atomics are used during lookup.

    Args:
        key: Packed ``int64`` cell key (see :func:`pack_key`).
        hash_slot: Hash table storing occupied-cell indices, ``-1`` if empty.
        cell_key: Per-occupied-cell full keys, indexed by the value in ``hash_slot``.
        hash_cap: Hash table capacity (a power of two).

    Returns:
        The occupied-cell slot index, or ``-1`` if the cell is empty/absent.
    """
    mask = hash_cap - 1
    h = wp.int32(hash64(wp.uint64(key)) & wp.uint64(mask))
    probe = wp.int32(0)
    while probe < hash_cap:
        c = hash_slot[h]
        if c == -1:
            return -1
        if cell_key[c] == key:
            return c
        h = (h + 1) & mask
        probe += 1
    return -1


# ---------------------------------------------------------------------------
# PC-index / query build kernels
# ---------------------------------------------------------------------------


@wp.kernel
def make_keys_kernel(
    coords: wp.array2d(dtype=wp.vec3),
    origin: wp.array(dtype=wp.vec3),
    inv_cell: wp.float32,
    bits: wp.int32,
    morton_bits: wp.int32,
    max_coord: wp.int32,
    M: wp.int32,
    keys: wp.array(dtype=wp.int64),
):
    """
    Compute a packed batch+Morton cell key for every point/query.

    Launched flat over ``B * M`` elements; the batch index is recovered as
    ``tid // M``. Cell coordinates are quantized with a shared per-batch
    ``origin`` and ``inv_cell = 1 / cell_size`` and clamped into
    ``[0, max_coord]``.

    Args:
        coords: Coordinates of shape ``(B, M)`` (Warp ``vec3`` array).
        origin: Per-batch grid origin (lower corner), shape ``(B,)``.
        inv_cell: Reciprocal of the cell size.
        bits: Number of Morton bits per axis.
        morton_bits: Total Morton bit width (``3 * bits``).
        max_coord: Maximum cell coordinate per axis (``2**bits - 1``).
        M: Number of elements per batch (points or queries).
        keys: Output packed keys of shape ``(B * M,)``.
    """
    tid = wp.tid()
    b = tid // M
    i = tid % M
    pt = coords[b, i]
    o = origin[b]
    cx = wp.clamp(wp.int32(wp.floor((pt[0] - o[0]) * inv_cell)), 0, max_coord)
    cy = wp.clamp(wp.int32(wp.floor((pt[1] - o[1]) * inv_cell)), 0, max_coord)
    cz = wp.clamp(wp.int32(wp.floor((pt[2] - o[2]) * inv_cell)), 0, max_coord)
    keys[tid] = pack_key(b, cx, cy, cz, bits, morton_bits)


@wp.kernel
def gather_pc_kernel(
    points: wp.array2d(dtype=wp.vec3),
    perm: wp.array(dtype=wp.int32),
    P: wp.int32,
    pc_orig: wp.array(dtype=wp.int32),
    pc_xyz4: wp.array2d(dtype=wp.float32),
    pc_norm: wp.array(dtype=wp.float32),
):
    """
    Gather the Morton-sorted point arrays from the sort permutation.

    For each sorted slot ``si``, ``perm[si]`` is the original flat point index;
    this writes the local point index, the padded ``xyz4`` coordinates, and the
    squared norm used by the GEMM distance formula.

    Args:
        points: Reference points of shape ``(B, P)`` (Warp ``vec3`` array).
        perm: Sort permutation (original flat indices), shape ``(B * P,)``.
        P: Number of points per batch.
        pc_orig: Output local point index per sorted slot.
        pc_xyz4: Output sorted coordinates ``(x, y, z, 0)``, shape ``(>=B*P, 4)``.
        pc_norm: Output squared norm ``x^2 + y^2 + z^2`` per sorted slot.
    """
    si = wp.tid()
    flat = perm[si]
    b = flat // P
    p = flat % P
    pt = points[b, p]
    pc_orig[si] = p
    pc_xyz4[si, 0] = pt[0]
    pc_xyz4[si, 1] = pt[1]
    pc_xyz4[si, 2] = pt[2]
    pc_xyz4[si, 3] = 0.0
    pc_norm[si] = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]


@wp.kernel
def gather_q_kernel(
    queries: wp.array2d(dtype=wp.vec3),
    perm: wp.array(dtype=wp.int32),
    Q: wp.int32,
    origin: wp.array(dtype=wp.vec3),
    inv_cell: wp.float32,
    max_coord: wp.int32,
    q_orig: wp.array(dtype=wp.int32),
    qp_xyz4: wp.array2d(dtype=wp.float32),
    qp_norm: wp.array(dtype=wp.float32),
    qcell_b: wp.array(dtype=wp.int32),
    qcell_cx: wp.array(dtype=wp.int32),
    qcell_cy: wp.array(dtype=wp.int32),
    qcell_cz: wp.array(dtype=wp.int32),
):
    """
    Gather the Morton-sorted query arrays and their integer cell coordinates.

    ``q_orig`` stores the *original* flat query id (``b * Q + q``) so search
    kernels can write results back in the caller's query order. The per-query
    cell coordinates are recomputed here (with the shared PC ``origin`` /
    ``inv_cell``) so no host-side Morton decode is needed downstream.

    Args:
        queries: Query points of shape ``(B, Q)`` (Warp ``vec3`` array).
        perm: Sort permutation (original flat query indices), shape ``(B * Q,)``.
        Q: Number of queries per batch.
        origin: Per-batch grid origin shared with the PC index, shape ``(B,)``.
        inv_cell: Reciprocal of the cell size.
        max_coord: Maximum cell coordinate per axis (``2**bits - 1``).
        q_orig: Output original flat query id per sorted slot.
        qp_xyz4: Output sorted query coordinates ``(x, y, z, 0)``.
        qp_norm: Output squared norm per sorted query.
        qcell_b: Output batch index per sorted query.
        qcell_cx: Output cell coordinate x per sorted query.
        qcell_cy: Output cell coordinate y per sorted query.
        qcell_cz: Output cell coordinate z per sorted query.
    """
    si = wp.tid()
    flat = perm[si]
    b = flat // Q
    q = flat % Q
    pt = queries[b, q]
    o = origin[b]
    q_orig[si] = flat
    qp_xyz4[si, 0] = pt[0]
    qp_xyz4[si, 1] = pt[1]
    qp_xyz4[si, 2] = pt[2]
    qp_xyz4[si, 3] = 0.0
    qp_norm[si] = pt[0] * pt[0] + pt[1] * pt[1] + pt[2] * pt[2]
    qcell_b[si] = b
    qcell_cx[si] = wp.clamp(wp.int32(wp.floor((pt[0] - o[0]) * inv_cell)), 0, max_coord)
    qcell_cy[si] = wp.clamp(wp.int32(wp.floor((pt[1] - o[1]) * inv_cell)), 0, max_coord)
    qcell_cz[si] = wp.clamp(wp.int32(wp.floor((pt[2] - o[2]) * inv_cell)), 0, max_coord)


@wp.kernel
def build_cell_hash_kernel(
    cell_key: wp.array(dtype=wp.int64),
    hash_cap: wp.int32,
    hash_slot: wp.array(dtype=wp.int32),
):
    """
    Insert every occupied cell into the open-addressed hash table.

    Each thread claims an empty slot for its cell index ``c`` using an ``int32``
    compare-and-swap on ``hash_slot`` (initialized to ``-1`` by the host), then
    linear-probes on collision. Only ``int32`` CAS is required because the full
    64-bit key lives in ``cell_key`` and is verified during lookup.

    Args:
        cell_key: Per-occupied-cell packed keys, shape ``(num_cells,)``.
        hash_cap: Hash table capacity (a power of two).
        hash_slot: Hash table to fill; ``-1`` means empty.
    """
    c = wp.tid()
    key = cell_key[c]
    mask = hash_cap - 1
    h = wp.int32(hash64(wp.uint64(key)) & wp.uint64(mask))
    probe = wp.int32(0)
    while probe < hash_cap:
        old = wp.atomic_cas(hash_slot, h, wp.int32(-1), c)
        if old == -1:
            return
        h = (h + 1) & mask
        probe += 1


# ---------------------------------------------------------------------------
# Tiled-task generation kernels (shared by B1 and B2)
# ---------------------------------------------------------------------------


@wp.kernel
def count_tasks_kernel(
    q_tile_b: wp.array(dtype=wp.int32),
    q_tile_cx: wp.array(dtype=wp.int32),
    q_tile_cy: wp.array(dtype=wp.int32),
    q_tile_cz: wp.array(dtype=wp.int32),
    offsets: wp.array2d(dtype=wp.int32),
    K: wp.int32,
    max_coord: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
    cell_key: wp.array(dtype=wp.int64),
    cell_start: wp.array(dtype=wp.int32),
    cell_end: wp.array(dtype=wp.int32),
    hash_slot: wp.array(dtype=wp.int32),
    hash_cap: wp.int32,
    tile_p: wp.int32,
    task_count: wp.array(dtype=wp.int32),
):
    """
    Count the number of ``(query-tile, point-chunk)`` tasks per query tile.

    For each query tile, scans the precomputed neighbor-cell offsets, looks up
    each non-empty PC cell, and accumulates ``ceil(cell_len / tile_p)`` chunks.

    Args:
        q_tile_b: Per-query-tile batch index.
        q_tile_cx: Per-query-tile base cell coordinate x.
        q_tile_cy: Per-query-tile base cell coordinate y.
        q_tile_cz: Per-query-tile base cell coordinate z.
        offsets: Neighbor-cell offset table of shape ``(K, 3)``.
        K: Number of neighbor offsets.
        max_coord: Maximum cell coordinate per axis.
        bits: Number of Morton bits per axis.
        morton_bits: Total Morton bit width.
        cell_key: Per-occupied-cell keys.
        cell_start: Per-occupied-cell start index into the sorted PC arrays.
        cell_end: Per-occupied-cell end index (exclusive).
        hash_slot: Hash table of occupied-cell indices.
        hash_cap: Hash table capacity.
        tile_p: Point-chunk width (``TILE_P``).
        task_count: Output task count per query tile.
    """
    qt = wp.tid()
    b = q_tile_b[qt]
    cx = q_tile_cx[qt]
    cy = q_tile_cy[qt]
    cz = q_tile_cz[qt]
    total = wp.int32(0)
    for k in range(K):
        ncx = cx + offsets[k, 0]
        ncy = cy + offsets[k, 1]
        ncz = cz + offsets[k, 2]
        if ncx < 0 or ncx > max_coord or ncy < 0 or ncy > max_coord or ncz < 0 or ncz > max_coord:
            continue
        nkey = pack_key(b, ncx, ncy, ncz, bits, morton_bits)
        c = lookup_cell(nkey, hash_slot, cell_key, hash_cap)
        if c >= 0:
            plen = cell_end[c] - cell_start[c]
            total += (plen + tile_p - 1) // tile_p
    task_count[qt] = total


@wp.kernel
def fill_tasks_kernel(
    q_tile_begin: wp.array(dtype=wp.int32),
    q_tile_count: wp.array(dtype=wp.int32),
    q_tile_b: wp.array(dtype=wp.int32),
    q_tile_cx: wp.array(dtype=wp.int32),
    q_tile_cy: wp.array(dtype=wp.int32),
    q_tile_cz: wp.array(dtype=wp.int32),
    task_offset: wp.array(dtype=wp.int32),
    offsets: wp.array2d(dtype=wp.int32),
    K: wp.int32,
    max_coord: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
    cell_key: wp.array(dtype=wp.int64),
    cell_start: wp.array(dtype=wp.int32),
    cell_end: wp.array(dtype=wp.int32),
    hash_slot: wp.array(dtype=wp.int32),
    hash_cap: wp.int32,
    tile_p: wp.int32,
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    task_p_begin: wp.array(dtype=wp.int32),
    task_p_count: wp.array(dtype=wp.int32),
):
    """
    Materialize the task list produced by :func:`count_tasks_kernel`.

    Each query tile owns the contiguous output range starting at
    ``task_offset[qt]`` (an exclusive scan of the per-tile counts), so writes are
    sequential and need no atomics.

    Args:
        q_tile_begin: Per-query-tile start index into the sorted query arrays.
        q_tile_count: Per-query-tile query count (``<= TILE_Q``).
        q_tile_b: Per-query-tile batch index.
        q_tile_cx: Per-query-tile base cell coordinate x.
        q_tile_cy: Per-query-tile base cell coordinate y.
        q_tile_cz: Per-query-tile base cell coordinate z.
        task_offset: Exclusive scan of per-tile task counts.
        offsets: Neighbor-cell offset table of shape ``(K, 3)``.
        K: Number of neighbor offsets.
        max_coord: Maximum cell coordinate per axis.
        bits: Number of Morton bits per axis.
        morton_bits: Total Morton bit width.
        cell_key: Per-occupied-cell keys.
        cell_start: Per-occupied-cell start index into the sorted PC arrays.
        cell_end: Per-occupied-cell end index (exclusive).
        hash_slot: Hash table of occupied-cell indices.
        hash_cap: Hash table capacity.
        tile_p: Point-chunk width (``TILE_P``).
        task_q_begin: Output query-tile start per task.
        task_q_count: Output query count per task.
        task_p_begin: Output point-chunk start per task.
        task_p_count: Output point-chunk length per task.
    """
    qt = wp.tid()
    out = task_offset[qt]
    qb = q_tile_begin[qt]
    qc = q_tile_count[qt]
    b = q_tile_b[qt]
    cx = q_tile_cx[qt]
    cy = q_tile_cy[qt]
    cz = q_tile_cz[qt]
    for k in range(K):
        ncx = cx + offsets[k, 0]
        ncy = cy + offsets[k, 1]
        ncz = cz + offsets[k, 2]
        if ncx < 0 or ncx > max_coord or ncy < 0 or ncy > max_coord or ncz < 0 or ncz > max_coord:
            continue
        nkey = pack_key(b, ncx, ncy, ncz, bits, morton_bits)
        c = lookup_cell(nkey, hash_slot, cell_key, hash_cap)
        if c < 0:
            continue
        ps = cell_start[c]
        pe = cell_end[c]
        pb = ps
        while pb < pe:
            pcount = wp.min(tile_p, pe - pb)
            task_q_begin[out] = qb
            task_q_count[out] = qc
            task_p_begin[out] = pb
            task_p_count[out] = pcount
            out += 1
            pb += tile_p


# ---------------------------------------------------------------------------
# Function A: scalar-hash search
# ---------------------------------------------------------------------------


@wp.kernel
def scalar_search_kernel(
    qp_xyz4: wp.array2d(dtype=wp.float32),
    q_orig: wp.array(dtype=wp.int32),
    qcell_b: wp.array(dtype=wp.int32),
    qcell_cx: wp.array(dtype=wp.int32),
    qcell_cy: wp.array(dtype=wp.int32),
    qcell_cz: wp.array(dtype=wp.int32),
    offsets: wp.array2d(dtype=wp.int32),
    K: wp.int32,
    max_coord: wp.int32,
    bits: wp.int32,
    morton_bits: wp.int32,
    cell_key: wp.array(dtype=wp.int64),
    cell_start: wp.array(dtype=wp.int32),
    cell_end: wp.array(dtype=wp.int32),
    hash_slot: wp.array(dtype=wp.int32),
    hash_cap: wp.int32,
    pc_xyz4: wp.array2d(dtype=wp.float32),
    pc_orig: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """
    Scalar-hash radius search (Function A): one thread per sorted query.

    Scans the precomputed near-to-far neighbor cells, looks up each via the hash
    table, and tests the contiguous sorted points in ``[cell_start, cell_end)``.
    Up to ``max_points`` first-found neighbors (in scan order) are written; the
    near-to-far ordering makes the early exit favor closer cells. Results are
    written at the original flat query id, so output query order matches input.

    Args:
        qp_xyz4: Sorted query coordinates ``(x, y, z, 0)``.
        q_orig: Original flat query id per sorted query.
        qcell_b: Batch index per sorted query.
        qcell_cx: Cell coordinate x per sorted query.
        qcell_cy: Cell coordinate y per sorted query.
        qcell_cz: Cell coordinate z per sorted query.
        offsets: Near-to-far neighbor-cell offsets, shape ``(K, 3)``.
        K: Number of neighbor offsets.
        max_coord: Maximum cell coordinate per axis.
        bits: Number of Morton bits per axis.
        morton_bits: Total Morton bit width.
        cell_key: Per-occupied-cell keys.
        cell_start: Per-occupied-cell start index into the sorted PC arrays.
        cell_end: Per-occupied-cell end index (exclusive).
        hash_slot: Hash table of occupied-cell indices.
        hash_cap: Hash table capacity.
        pc_xyz4: Sorted point coordinates ``(x, y, z, 0)``.
        pc_orig: Local point index per sorted point.
        radius2: Squared search radius.
        max_points: Maximum neighbors to record per query.
        return_dists: Whether to write distances.
        return_points: Whether to write neighbor coordinates.
        out_idx: Output neighbor indices, shape ``(B * Q, max_points)``.
        out_count: Output neighbor count per query, shape ``(B * Q,)``.
        out_dist: Output distances (only written when ``return_dists``).
        out_pts: Output neighbor coordinates (only written when ``return_points``).
    """
    sq = wp.tid()
    flat_q = q_orig[sq]
    qx = qp_xyz4[sq, 0]
    qy = qp_xyz4[sq, 1]
    qz = qp_xyz4[sq, 2]
    b = qcell_b[sq]
    cx = qcell_cx[sq]
    cy = qcell_cy[sq]
    cz = qcell_cz[sq]

    count = wp.int32(0)
    for k in range(K):
        if count >= max_points:
            break
        ncx = cx + offsets[k, 0]
        ncy = cy + offsets[k, 1]
        ncz = cz + offsets[k, 2]
        if ncx < 0 or ncx > max_coord or ncy < 0 or ncy > max_coord or ncz < 0 or ncz > max_coord:
            continue
        nkey = pack_key(b, ncx, ncy, ncz, bits, morton_bits)
        c = lookup_cell(nkey, hash_slot, cell_key, hash_cap)
        if c < 0:
            continue
        s = cell_start[c]
        e = cell_end[c]
        for si in range(s, e):
            px = pc_xyz4[si, 0]
            py = pc_xyz4[si, 1]
            pz = pc_xyz4[si, 2]
            dx = px - qx
            dy = py - qy
            dz = pz - qz
            d2 = dx * dx + dy * dy + dz * dz
            if d2 <= radius2:
                out_idx[flat_q, count] = pc_orig[si]
                if return_dists:
                    out_dist[flat_q, count] = wp.sqrt(d2)
                if return_points:
                    out_pts[flat_q, count] = wp.vec3(px, py, pz)
                count += 1
                if count >= max_points:
                    break
    out_count[flat_q] = count


# ---------------------------------------------------------------------------
# Function B1: tiled direct-FMA search
# ---------------------------------------------------------------------------


@wp.kernel
def tiled_fma_kernel(
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    task_p_begin: wp.array(dtype=wp.int32),
    task_p_count: wp.array(dtype=wp.int32),
    qp_xyz4: wp.array2d(dtype=wp.float32),
    q_orig: wp.array(dtype=wp.int32),
    pc_xyz4: wp.array2d(dtype=wp.float32),
    pc_orig: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """
    Tiled direct-FMA radius search (Function B1).

    Launched with a 2D ``wp.launch(dim=(num_tasks, TILE_P))``: the first index is
    the task and the second is the lane (one candidate point per lane). Each lane
    computes its candidate's direct squared distance to every query in the task's
    ``TILE_Q`` tile and appends in-radius hits via an atomic per-query slot
    counter (``out_count``, which may overshoot ``max_points``; the host clamps it
    afterward). Queries are read from global memory (same-address broadcast across
    lanes), so no cooperative tile load is needed; this keeps the design's
    one-thread-per-candidate, loop-over-``TILE_Q`` mapping while staying robust.

    Args:
        task_q_begin: Per-task query-tile start index.
        task_q_count: Per-task query count (``<= TILE_Q``).
        task_p_begin: Per-task point-chunk start index.
        task_p_count: Per-task point-chunk length (``<= TILE_P``).
        qp_xyz4: Sorted query coordinates ``(x, y, z, 0)``.
        q_orig: Original flat query id per sorted query.
        pc_xyz4: Sorted point coordinates ``(x, y, z, 0)``.
        pc_orig: Local point index per sorted point.
        radius2: Squared search radius.
        max_points: Maximum neighbors to record per query.
        return_dists: Whether to write distances.
        return_points: Whether to write neighbor coordinates.
        out_idx: Output neighbor indices, shape ``(B * Q, max_points)``.
        out_count: Atomic per-query slot counter / neighbor count.
        out_dist: Output distances (only written when ``return_dists``).
        out_pts: Output neighbor coordinates (only written when ``return_points``).
    """
    task_id, lane = wp.tid()
    pcount = task_p_count[task_id]
    if lane >= pcount:
        return

    qb = task_q_begin[task_id]
    qc = task_q_count[task_id]

    # Early out: skip the whole task if every query in its tile already has
    # max_points neighbors. These are cheap non-atomic reads; the atomic below
    # still guarantees correctness. For large radii (dense cells) this lets later
    # task waves stop scanning once queries fill, avoiding near-brute-force cost.
    all_full = True
    for iq in range(TILE_Q):
        if iq < qc:
            if out_count[q_orig[qb + iq]] < max_points:
                all_full = False
    if all_full:
        return

    pb = task_p_begin[task_id]
    sp = pb + lane

    px = pc_xyz4[sp, 0]
    py = pc_xyz4[sp, 1]
    pz = pc_xyz4[sp, 2]
    porig = pc_orig[sp]

    for iq in range(TILE_Q):
        if iq < qc:
            flat_q = q_orig[qb + iq]
            # Per-query early out: don't compute distances for a full query.
            if out_count[flat_q] < max_points:
                qx = qp_xyz4[qb + iq, 0]
                qy = qp_xyz4[qb + iq, 1]
                qz = qp_xyz4[qb + iq, 2]
                dx = px - qx
                dy = py - qy
                dz = pz - qz
                d2 = dx * dx + dy * dy + dz * dz
                if d2 <= radius2:
                    slot = wp.atomic_add(out_count, flat_q, 1)
                    if slot < max_points:
                        out_idx[flat_q, slot] = porig
                        if return_dists:
                            out_dist[flat_q, slot] = wp.sqrt(d2)
                        if return_points:
                            out_pts[flat_q, slot] = wp.vec3(px, py, pz)


# ---------------------------------------------------------------------------
# Function B2: tiled GEMM search
# ---------------------------------------------------------------------------


@wp.kernel
def tiled_gemm_kernel(
    task_q_begin: wp.array(dtype=wp.int32),
    task_q_count: wp.array(dtype=wp.int32),
    task_p_begin: wp.array(dtype=wp.int32),
    task_p_count: wp.array(dtype=wp.int32),
    lane_iota: wp.array(dtype=wp.int32),
    qp_xyz4: wp.array2d(dtype=wp.float32),
    qp_norm: wp.array(dtype=wp.float32),
    q_orig: wp.array(dtype=wp.int32),
    pc_xyz4: wp.array2d(dtype=wp.float32),
    pc_norm: wp.array(dtype=wp.float32),
    pc_orig: wp.array(dtype=wp.int32),
    radius2: wp.float32,
    max_points: wp.int32,
    return_dists: wp.bool,
    return_points: wp.bool,
    out_idx: wp.array2d(dtype=wp.int32),
    out_count: wp.array(dtype=wp.int32),
    out_dist: wp.array2d(dtype=wp.float32),
    out_pts: wp.array2d(dtype=wp.vec3),
):
    """
    Tiled GEMM radius search (Function B2).

    Launched with ``wp.launch_tiled(dim=num_tasks, block_dim=TILE_P)``: one block
    per task, one lane per candidate point. Distances use the expansion
    :math:`D^2 = \\lVert Q \\rVert^2 + \\lVert P \\rVert^2 - 2 Q P^T`. The ``Q``
    and ``P`` tiles (padded to width 4) are multiplied with ``wp.tile_matmul``
    into a ``(TILE_Q, TILE_P)`` tile; ``wp.untile`` then hands each lane (point)
    the ``TILE_Q`` dot products for its candidate, which are combined with the
    cached norms. In-radius hits append via the atomic per-query slot counter.

    Args:
        task_q_begin: Per-task query-tile start index.
        task_q_count: Per-task query count (``<= TILE_Q``).
        task_p_begin: Per-task point-chunk start index.
        task_p_count: Per-task point-chunk length (``<= TILE_P``).
        lane_iota: Constant ``[0, 1, ..., TILE_P - 1]`` used to derive lane ids.
        qp_xyz4: Sorted query coordinates ``(x, y, z, 0)``.
        qp_norm: Squared norm per sorted query.
        q_orig: Original flat query id per sorted query.
        pc_xyz4: Sorted point coordinates ``(x, y, z, 0)``.
        pc_norm: Squared norm per sorted point.
        pc_orig: Local point index per sorted point.
        radius2: Squared search radius.
        max_points: Maximum neighbors to record per query.
        return_dists: Whether to write distances.
        return_points: Whether to write neighbor coordinates.
        out_idx: Output neighbor indices, shape ``(B * Q, max_points)``.
        out_count: Atomic per-query slot counter / neighbor count.
        out_dist: Output distances (only written when ``return_dists``).
        out_pts: Output neighbor coordinates (only written when ``return_points``).
    """
    task_id = wp.tid()
    qb = task_q_begin[task_id]
    qc = task_q_count[task_id]
    pb = task_p_begin[task_id]
    pcount = task_p_count[task_id]

    # Collective tile ops first (no divergence): load Q/P tiles, compute the
    # query-by-point dot-product tile, and distribute one point's column per lane.
    # Shared storage is used for the matmul operands per the design, and the
    # output uses the 3-arg accumulating tile_matmul into a zero-initialized tile.
    q_tile = wp.tile_load(qp_xyz4, shape=(TILE_Q, 4), offset=(qb, 0), storage="shared")
    p_tile = wp.tile_load(pc_xyz4, shape=(TILE_P, 4), offset=(pb, 0), storage="shared")
    pt_tile = wp.tile_transpose(p_tile)  # (4, TILE_P)
    dot_tile = wp.tile_zeros(shape=(TILE_Q, TILE_P), dtype=wp.float32)
    wp.tile_matmul(q_tile, pt_tile, dot_tile)  # dot_tile = q_tile @ pt_tile
    # untile distributes the last tile dimension (TILE_P == block_dim) across
    # lanes, so each lane (one candidate point) receives its TILE_Q dot products.
    dot = wp.untile(dot_tile)
    lane = wp.untile(wp.tile_load(lane_iota, shape=(TILE_P,)))

    valid = lane < pcount
    sp = pb + lane
    p_norm = float(0.0)
    porig = wp.int32(-1)
    px = float(0.0)
    py = float(0.0)
    pz = float(0.0)
    if valid:
        p_norm = pc_norm[sp]
        porig = pc_orig[sp]
        if return_points:
            px = pc_xyz4[sp, 0]
            py = pc_xyz4[sp, 1]
            pz = pc_xyz4[sp, 2]

    for iq in range(TILE_Q):
        if iq < qc:
            if valid:
                flat_q = q_orig[qb + iq]
                # Per-query early out: skip already-full queries (trims atomics).
                # Note the tile_matmul above still runs for every task, so GEMM
                # cannot early-terminate the scan like the scalar/FMA paths.
                if out_count[flat_q] < max_points:
                    q_norm = qp_norm[qb + iq]
                    d2 = q_norm + p_norm - 2.0 * dot[iq]
                    if d2 <= radius2:
                        slot = wp.atomic_add(out_count, flat_q, 1)
                        if slot < max_points:
                            out_idx[flat_q, slot] = porig
                            if return_dists:
                                out_dist[flat_q, slot] = wp.sqrt(wp.max(d2, 0.0))
                            if return_points:
                                out_pts[flat_q, slot] = wp.vec3(px, py, pz)
