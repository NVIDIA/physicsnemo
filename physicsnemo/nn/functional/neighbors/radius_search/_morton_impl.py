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
Torch <-> Warp host glue for the Morton-grid radius search backend.

This module builds a reusable :class:`MortonGridIndex` from a point cloud
(Morton-sorted grid + open-addressed cell hash) and runs the three search
variants from the design:

* ``scalar`` -- :func:`radius_search_morton_scalar_hash` (Function A)
* ``fma``    -- :func:`radius_search_morton_tiled_fma` (Function B1)
* ``gemm``   -- :func:`radius_search_morton_tiled_gemm` (Function B2)

:func:`radius_search_morton` is the single entry point used by the dispatch in
``_warp_impl.py``. It mirrors the return contract of the ``max_points`` path of
``radius_search_impl``: ``(indices, points, distances, num_neighbors)`` with the
batch dimension squeezed for unbatched inputs and outputs cast back to the input
dtype.

Implementation notes:

* Sorting and prefix sums are done with ``torch`` (``argsort`` / ``cumsum``)
  rather than ``wp.utils.radix_sort_pairs`` / ``wp.utils.array_scan``. This keeps
  the backend robust across Warp versions and matches the existing tiled-BVH
  path, which already Morton-orders via ``torch.argsort``. The sort still
  produces the Morton ordering the design relies on.
* Cell lookup uses the custom open-addressed hash table built with
  ``wp.atomic_cas`` (see ``_morton_kernels.lookup_cell``). The unique cell keys
  are also stored sorted, so a binary search is a drop-in fallback if a build
  lacks ``wp.atomic_cas``.
* This backend is CUDA-only; the caller falls back to the hash-grid path on CPU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import warp as wp

from physicsnemo.core.function_spec import FunctionSpec

from . import _morton_kernels as mk
from .utils import validate_inputs

TILE_Q = mk.TILE_Q
TILE_P = mk.TILE_P

VARIANTS = ("scalar", "fma", "gemm")


def _next_pow2(value: int) -> int:
    """Return the smallest power of two that is ``>= value`` (and ``>= 1``)."""
    if value <= 1:
        return 1
    return 1 << (value - 1).bit_length()


def _exclusive_scan(counts: torch.Tensor) -> torch.Tensor:
    """Exclusive prefix sum of a 1D integer tensor, returned as ``int64``."""
    counts64 = counts.to(torch.int64)
    return torch.cumsum(counts64, dim=0) - counts64


def _neighbor_offsets(cell_size_factor: float, device: torch.device) -> torch.Tensor:
    """
    Build the near-to-far neighbor-cell offset table.

    With ``cell_size = radius * cell_size_factor`` a query must look ``R =
    ceil(1 / cell_size_factor)`` cells out along each axis, giving a
    ``(2R + 1)^3`` neighborhood (27 cells for factor 1, 125 for factor 0.5).
    Offsets are sorted by squared magnitude so the scalar kernel's early exit
    favors closer cells.

    Args:
        cell_size_factor: ``cell_size / radius`` ratio.
        device: Device for the returned tensor.

    Returns:
        An ``(K, 3)`` int32 tensor of cell offsets ordered near to far.
    """
    R = int(math.ceil(1.0 / cell_size_factor))
    rng = range(-R, R + 1)
    offs = [(dx, dy, dz) for dx in rng for dy in rng for dz in rng]
    offs.sort(key=lambda o: o[0] * o[0] + o[1] * o[1] + o[2] * o[2])
    return torch.tensor(offs, dtype=torch.int32, device=device)


@dataclass
class MortonGridIndex:
    """
    Reusable PC-side Morton grid + open-addressed hash index.

    Holds the Morton-sorted point arrays, the per-occupied-cell ranges, the cell
    hash table, and the grid metadata needed to quantize queries consistently.
    Built once per :func:`radius_search_morton` call and shared by all three
    search variants.

    Attributes:
        B: Batch size.
        P: Points per batch.
        Np: Total points (``B * P``).
        cell_size: Grid cell size (``radius * cell_size_factor``).
        radius: Search radius.
        bits_per_axis: Morton bits per axis.
        morton_bits: Total Morton bit width (``3 * bits_per_axis``).
        max_coord: Maximum cell coordinate per axis (``2**bits_per_axis - 1``).
        hash_cap: Hash table capacity (a power of two).
        num_cells: Number of occupied cells.
        origin: Per-batch grid origin, shape ``(B, 3)`` float32.
        offsets: Near-to-far neighbor offsets, shape ``(K, 3)`` int32.
        pc_xyz4: Sorted point coordinates ``(x, y, z, 0)``, padded for tiling.
        pc_norm: Squared norm per sorted point.
        pc_orig: Local point index per sorted point.
        cell_key: Per-occupied-cell packed keys (sorted ascending).
        cell_start: Per-occupied-cell start index into the sorted arrays.
        cell_end: Per-occupied-cell end index (exclusive).
        hash_slot: Open-addressed hash table of occupied-cell indices.
    """

    B: int
    P: int
    Np: int
    cell_size: float
    radius: float
    bits_per_axis: int
    morton_bits: int
    max_coord: int
    hash_cap: int
    num_cells: int
    origin: torch.Tensor
    offsets: torch.Tensor
    pc_xyz4: torch.Tensor
    pc_norm: torch.Tensor
    pc_orig: torch.Tensor
    cell_key: torch.Tensor
    cell_start: torch.Tensor
    cell_end: torch.Tensor
    hash_slot: torch.Tensor


def build_morton_pc_index(
    points: torch.Tensor,
    radius: float,
    cell_size_factor: float,
    wp_device,
    wp_stream,
) -> MortonGridIndex:
    """
    Build the reusable Morton grid index for a point cloud.

    Quantizes points to a uniform grid (per-batch origin padded one cell below
    the bounding box so query cells stay non-negative), Morton-sorts them, builds
    contiguous per-occupied-cell ranges, and inserts the cells into the
    open-addressed hash table.

    Args:
        points: Reference points of shape ``(B, N, 3)``, float32, CUDA.
        radius: Search radius.
        cell_size_factor: ``cell_size / radius`` ratio (1.0 -> 27-cell stencil).
        wp_device: Warp launch device (from ``warp_launch_context``).
        wp_stream: Warp launch stream (from ``warp_launch_context``).

    Returns:
        The populated :class:`MortonGridIndex`.

    Raises:
        ValueError: If the grid extent plus batch bits exceed the positive
            ``int64`` key range (suggesting a larger ``radius`` / cell size or
            per-batch processing).
    """
    device = points.device
    B, N, _ = points.shape
    P = N
    Np = B * N

    # Inflate the grid cell slightly relative to the nominal radius*factor. The
    # neighbor stencil is R = ceil(1/cell_size_factor) cells; for correctness a
    # neighbor within `radius` must land within R cells, i.e. radius/cell_size
    # must stay <= R. With cell_size == radius (factor 1) floating-point in the
    # floor() quantization makes radius*inv_cell ~1.0000001 > 1.0, so a boundary
    # neighbor can fall 2 cells away and be missed. A ~0.1% larger cell keeps
    # radius/cell_size strictly below R with margin >> fp epsilon.
    cell_size = float(radius) * float(cell_size_factor) * (1.0 + 1e-3)
    inv_cell = 1.0 / cell_size

    pmin = points.amin(dim=1)
    pmax = points.amax(dim=1)
    origin = (pmin - cell_size).contiguous()
    extent = pmax - origin + cell_size
    n_cells = int(torch.ceil(extent / cell_size).max().item())
    n_cells = max(n_cells, 1)
    bits_per_axis = max(1, n_cells.bit_length())
    batch_bits = 0 if B <= 1 else (B - 1).bit_length()
    morton_bits = 3 * bits_per_axis
    if morton_bits + batch_bits > 63:
        raise ValueError(
            "Morton key would overflow int64: "
            f"{morton_bits} morton bits + {batch_bits} batch bits > 63. "
            "Use a larger radius/cell_size or process fewer batches per call."
        )
    max_coord = (1 << bits_per_axis) - 1

    points_wp = wp.from_torch(points, dtype=wp.vec3, return_ctype=True)
    origin_wp = wp.from_torch(origin, dtype=wp.vec3, return_ctype=True)

    keys = torch.empty(Np, dtype=torch.int64, device=device)
    wp.launch(
        kernel=mk.make_keys_kernel,
        dim=Np,
        inputs=[
            points_wp,
            origin_wp,
            float(inv_cell),
            int(bits_per_axis),
            int(morton_bits),
            int(max_coord),
            int(P),
            wp.from_torch(keys, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    # Stability is not required: we only group points by equal cell key, so any
    # ordering within a cell is fine. The non-stable sort is faster on CUDA.
    perm = torch.argsort(keys)
    keys_sorted = keys[perm]
    perm32 = perm.to(torch.int32).contiguous()

    n_pad = Np + TILE_P
    pc_xyz4 = torch.zeros((n_pad, 4), dtype=torch.float32, device=device)
    pc_norm = torch.zeros(n_pad, dtype=torch.float32, device=device)
    pc_orig = torch.zeros(n_pad, dtype=torch.int32, device=device)
    wp.launch(
        kernel=mk.gather_pc_kernel,
        dim=Np,
        inputs=[
            points_wp,
            wp.from_torch(perm32, return_ctype=True),
            int(P),
            wp.from_torch(pc_orig, return_ctype=True),
            wp.from_torch(pc_xyz4, return_ctype=True),
            wp.from_torch(pc_norm, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    # Unique occupied-cell ranges over the Morton-sorted keys (torch-side; the
    # resulting cell_key array is ascending, which also enables a binary-search
    # fallback for cell lookup).
    flags = torch.ones(Np, dtype=torch.bool, device=device)
    flags[1:] = keys_sorted[1:] != keys_sorted[:-1]
    starts = torch.nonzero(flags, as_tuple=False).flatten()
    num_cells = int(starts.numel())
    cell_key = keys_sorted[starts].contiguous()
    cell_start = starts.to(torch.int32).contiguous()
    ends = torch.empty_like(starts)
    if num_cells > 1:
        ends[:-1] = starts[1:]
    ends[-1] = Np
    cell_end = ends.to(torch.int32).contiguous()

    hash_cap = _next_pow2(max(8, 4 * num_cells))
    hash_slot = torch.full((hash_cap,), -1, dtype=torch.int32, device=device)
    wp.launch(
        kernel=mk.build_cell_hash_kernel,
        dim=num_cells,
        inputs=[
            wp.from_torch(cell_key, return_ctype=True),
            int(hash_cap),
            wp.from_torch(hash_slot, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    return MortonGridIndex(
        B=B,
        P=P,
        Np=Np,
        cell_size=cell_size,
        radius=float(radius),
        bits_per_axis=int(bits_per_axis),
        morton_bits=int(morton_bits),
        max_coord=int(max_coord),
        hash_cap=int(hash_cap),
        num_cells=num_cells,
        origin=origin,
        offsets=_neighbor_offsets(cell_size_factor, device),
        pc_xyz4=pc_xyz4,
        pc_norm=pc_norm,
        pc_orig=pc_orig,
        cell_key=cell_key,
        cell_start=cell_start,
        cell_end=cell_end,
        hash_slot=hash_slot,
    )


def _prepare_queries(
    index: MortonGridIndex,
    queries: torch.Tensor,
    wp_device,
    wp_stream,
) -> dict:
    """
    Morton-sort the queries and gather their sorted arrays + cell coordinates.

    Uses the index's shared ``origin`` / ``cell_size`` so query cells align with
    the PC grid. Returns a dict of device tensors consumed by the search and
    task-building helpers.

    Args:
        index: The PC-side Morton index.
        queries: Query points of shape ``(B, Q, 3)``, float32, CUDA.
        wp_device: Warp launch device.
        wp_stream: Warp launch stream.

    Returns:
        Dict with sorted query arrays, per-query cell coordinates, and shapes.
    """
    device = queries.device
    B, Q, _ = queries.shape
    Bq = B * Q
    inv_cell = 1.0 / index.cell_size

    queries_wp = wp.from_torch(queries, dtype=wp.vec3, return_ctype=True)
    origin_wp = wp.from_torch(index.origin, dtype=wp.vec3, return_ctype=True)

    qkeys = torch.empty(Bq, dtype=torch.int64, device=device)
    wp.launch(
        kernel=mk.make_keys_kernel,
        dim=Bq,
        inputs=[
            queries_wp,
            origin_wp,
            float(inv_cell),
            int(index.bits_per_axis),
            int(index.morton_bits),
            int(index.max_coord),
            int(Q),
            wp.from_torch(qkeys, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    perm = torch.argsort(qkeys)
    qkeys_sorted = qkeys[perm]
    perm32 = perm.to(torch.int32).contiguous()

    bq_pad = Bq + TILE_Q
    qp_xyz4 = torch.zeros((bq_pad, 4), dtype=torch.float32, device=device)
    qp_norm = torch.zeros(bq_pad, dtype=torch.float32, device=device)
    q_orig = torch.zeros(bq_pad, dtype=torch.int32, device=device)
    qcell_b = torch.zeros(bq_pad, dtype=torch.int32, device=device)
    qcell_cx = torch.zeros(bq_pad, dtype=torch.int32, device=device)
    qcell_cy = torch.zeros(bq_pad, dtype=torch.int32, device=device)
    qcell_cz = torch.zeros(bq_pad, dtype=torch.int32, device=device)
    wp.launch(
        kernel=mk.gather_q_kernel,
        dim=Bq,
        inputs=[
            queries_wp,
            wp.from_torch(perm32, return_ctype=True),
            int(Q),
            origin_wp,
            float(inv_cell),
            int(index.max_coord),
            wp.from_torch(q_orig, return_ctype=True),
            wp.from_torch(qp_xyz4, return_ctype=True),
            wp.from_torch(qp_norm, return_ctype=True),
            wp.from_torch(qcell_b, return_ctype=True),
            wp.from_torch(qcell_cx, return_ctype=True),
            wp.from_torch(qcell_cy, return_ctype=True),
            wp.from_torch(qcell_cz, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    return {
        "B": B,
        "Q": Q,
        "Bq": Bq,
        "qkeys_sorted": qkeys_sorted,
        "qp_xyz4": qp_xyz4,
        "qp_norm": qp_norm,
        "q_orig": q_orig,
        "qcell_b": qcell_b,
        "qcell_cx": qcell_cx,
        "qcell_cy": qcell_cy,
        "qcell_cz": qcell_cz,
    }


def _build_tiles_and_tasks(
    index: MortonGridIndex,
    qprep: dict,
    wp_device,
    wp_stream,
) -> dict:
    """
    Build the query tiles and the ``(query-tile, point-chunk)`` task list.

    Splits each occupied query cell into ``TILE_Q`` chunks, then runs the
    two-pass (count -> exclusive scan -> fill) task generation so the tiled
    kernels can map one block per task.

    Args:
        index: The PC-side Morton index.
        qprep: The dict returned by :func:`_prepare_queries`.
        wp_device: Warp launch device.
        wp_stream: Warp launch stream.

    Returns:
        Dict with the task arrays and ``num_tasks``.
    """
    device = index.pc_xyz4.device
    Bq = qprep["Bq"]
    qkeys_sorted = qprep["qkeys_sorted"]

    # Occupied query-cell ranges (mirrors the PC-side construction).
    flags = torch.ones(Bq, dtype=torch.bool, device=device)
    flags[1:] = qkeys_sorted[1:] != qkeys_sorted[:-1]
    qstarts = torch.nonzero(flags, as_tuple=False).flatten()
    num_qcells = int(qstarts.numel())
    qends = torch.empty_like(qstarts)
    if num_qcells > 1:
        qends[:-1] = qstarts[1:]
    qends[-1] = Bq
    qcell_len = qends - qstarts

    tiles_per_cell = (qcell_len + TILE_Q - 1) // TILE_Q
    num_qtiles = int(tiles_per_cell.sum().item())
    tile_off = _exclusive_scan(tiles_per_cell)

    cell_of_tile = torch.repeat_interleave(
        torch.arange(num_qcells, device=device), tiles_per_cell
    )
    local_tile = torch.arange(num_qtiles, device=device) - tile_off[cell_of_tile]
    tile_begin = qstarts[cell_of_tile] + local_tile * TILE_Q
    q_tile_begin = tile_begin.to(torch.int32).contiguous()
    q_tile_count = (
        torch.clamp(qends[cell_of_tile] - tile_begin, max=TILE_Q)
        .to(torch.int32)
        .contiguous()
    )

    # Per-tile base cell coordinates (taken at each cell's first sorted query).
    cell_starts_long = qstarts
    q_tile_b = qprep["qcell_b"][cell_starts_long][cell_of_tile].to(torch.int32).contiguous()
    q_tile_cx = qprep["qcell_cx"][cell_starts_long][cell_of_tile].to(torch.int32).contiguous()
    q_tile_cy = qprep["qcell_cy"][cell_starts_long][cell_of_tile].to(torch.int32).contiguous()
    q_tile_cz = qprep["qcell_cz"][cell_starts_long][cell_of_tile].to(torch.int32).contiguous()

    offsets_wp = wp.from_torch(index.offsets, dtype=wp.int32, return_ctype=True)
    K = int(index.offsets.shape[0])

    task_count = torch.empty(num_qtiles, dtype=torch.int32, device=device)
    wp.launch(
        kernel=mk.count_tasks_kernel,
        dim=num_qtiles,
        inputs=[
            wp.from_torch(q_tile_b, return_ctype=True),
            wp.from_torch(q_tile_cx, return_ctype=True),
            wp.from_torch(q_tile_cy, return_ctype=True),
            wp.from_torch(q_tile_cz, return_ctype=True),
            offsets_wp,
            K,
            int(index.max_coord),
            int(index.bits_per_axis),
            int(index.morton_bits),
            wp.from_torch(index.cell_key, return_ctype=True),
            wp.from_torch(index.cell_start, return_ctype=True),
            wp.from_torch(index.cell_end, return_ctype=True),
            wp.from_torch(index.hash_slot, return_ctype=True),
            int(index.hash_cap),
            int(TILE_P),
            wp.from_torch(task_count, return_ctype=True),
        ],
        device=wp_device,
        stream=wp_stream,
    )

    task_offset = _exclusive_scan(task_count).to(torch.int32).contiguous()
    num_tasks = int(task_count.sum().item())

    task_q_begin = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_q_count = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_p_begin = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)
    task_p_count = torch.empty(max(num_tasks, 1), dtype=torch.int32, device=device)

    if num_tasks > 0:
        wp.launch(
            kernel=mk.fill_tasks_kernel,
            dim=num_qtiles,
            inputs=[
                wp.from_torch(q_tile_begin, return_ctype=True),
                wp.from_torch(q_tile_count, return_ctype=True),
                wp.from_torch(q_tile_b, return_ctype=True),
                wp.from_torch(q_tile_cx, return_ctype=True),
                wp.from_torch(q_tile_cy, return_ctype=True),
                wp.from_torch(q_tile_cz, return_ctype=True),
                wp.from_torch(task_offset, return_ctype=True),
                offsets_wp,
                K,
                int(index.max_coord),
                int(index.bits_per_axis),
                int(index.morton_bits),
                wp.from_torch(index.cell_key, return_ctype=True),
                wp.from_torch(index.cell_start, return_ctype=True),
                wp.from_torch(index.cell_end, return_ctype=True),
                wp.from_torch(index.hash_slot, return_ctype=True),
                int(index.hash_cap),
                int(TILE_P),
                wp.from_torch(task_q_begin, return_ctype=True),
                wp.from_torch(task_q_count, return_ctype=True),
                wp.from_torch(task_p_begin, return_ctype=True),
                wp.from_torch(task_p_count, return_ctype=True),
            ],
            device=wp_device,
            stream=wp_stream,
        )

    return {
        "num_tasks": num_tasks,
        "task_q_begin": task_q_begin,
        "task_q_count": task_q_count,
        "task_p_begin": task_p_begin,
        "task_p_count": task_p_count,
    }


def _alloc_outputs(
    B: int,
    Q: int,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    device: torch.device,
):
    """Allocate the output tensors for a search (indices/counts always; optional)."""
    out_idx = torch.zeros((B, Q, max_points), dtype=torch.int32, device=device)
    out_count = torch.zeros(B * Q, dtype=torch.int32, device=device)
    out_dist = (
        torch.zeros((B, Q, max_points), dtype=torch.float32, device=device)
        if return_dists
        else None
    )
    out_pts = (
        torch.zeros((B, Q, max_points, 3), dtype=torch.float32, device=device)
        if return_points
        else None
    )
    return out_idx, out_count, out_dist, out_pts


def _wp_out(out_idx, out_count, out_dist, out_pts, Bq, max_points, return_dists, return_points):
    """Build the Warp ctype views of the (flattened) output tensors."""
    idx_wp = wp.from_torch(out_idx.view(Bq, max_points), return_ctype=True)
    count_wp = wp.from_torch(out_count, return_ctype=True)
    dist_wp = (
        wp.from_torch(out_dist.view(Bq, max_points), return_ctype=True)
        if return_dists
        else None
    )
    pts_wp = (
        wp.from_torch(out_pts.view(Bq, max_points, 3), dtype=wp.vec3, return_ctype=True)
        if return_points
        else None
    )
    return idx_wp, count_wp, dist_wp, pts_wp


def radius_search_morton_scalar_hash(
    index: MortonGridIndex,
    qprep: dict,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """
    Function A: scalar-hash search (one thread per query).

    Args:
        index: The PC-side Morton index.
        qprep: The dict returned by :func:`_prepare_queries`.
        radius: Search radius.
        max_points: Maximum neighbors per query.
        return_dists: Whether to return distances.
        return_points: Whether to return neighbor coordinates.
        wp_device: Warp launch device.
        wp_stream: Warp launch stream.

    Returns:
        Tuple ``(indices, points, distances, num_neighbors)`` with batch dim
        intact (``(B, Q, ...)``); ``points``/``distances`` are ``None`` when not
        requested.
    """
    B, Q, Bq = qprep["B"], qprep["Q"], qprep["Bq"]
    device = index.pc_xyz4.device
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_out(
        out_idx, out_count, out_dist, out_pts, Bq, max_points, return_dists, return_points
    )

    wp.launch(
        kernel=mk.scalar_search_kernel,
        dim=Bq,
        inputs=[
            wp.from_torch(qprep["qp_xyz4"], return_ctype=True),
            wp.from_torch(qprep["q_orig"], return_ctype=True),
            wp.from_torch(qprep["qcell_b"], return_ctype=True),
            wp.from_torch(qprep["qcell_cx"], return_ctype=True),
            wp.from_torch(qprep["qcell_cy"], return_ctype=True),
            wp.from_torch(qprep["qcell_cz"], return_ctype=True),
            wp.from_torch(index.offsets, dtype=wp.int32, return_ctype=True),
            int(index.offsets.shape[0]),
            int(index.max_coord),
            int(index.bits_per_axis),
            int(index.morton_bits),
            wp.from_torch(index.cell_key, return_ctype=True),
            wp.from_torch(index.cell_start, return_ctype=True),
            wp.from_torch(index.cell_end, return_ctype=True),
            wp.from_torch(index.hash_slot, return_ctype=True),
            int(index.hash_cap),
            wp.from_torch(index.pc_xyz4, return_ctype=True),
            wp.from_torch(index.pc_orig, return_ctype=True),
            float(radius) * float(radius),
            int(max_points),
            bool(return_dists),
            bool(return_points),
            idx_wp,
            count_wp,
            dist_wp,
            pts_wp,
        ],
        device=wp_device,
        stream=wp_stream,
    )

    num = out_count.view(B, Q)
    return out_idx, out_pts, out_dist, num


def _run_tiled(
    index: MortonGridIndex,
    qprep: dict,
    tasks: dict,
    variant: str,
    radius: float,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    wp_device,
    wp_stream,
):
    """
    Shared launcher for the tiled FMA (B1) and GEMM (B2) variants.

    Both consume the same task list and write via an atomic per-query slot
    counter that can overshoot ``max_points``; the count is clamped afterward.

    Args:
        index: The PC-side Morton index.
        qprep: The dict returned by :func:`_prepare_queries`.
        tasks: The dict returned by :func:`_build_tiles_and_tasks`.
        variant: Either ``"fma"`` or ``"gemm"``.
        radius: Search radius.
        max_points: Maximum neighbors per query.
        return_dists: Whether to return distances.
        return_points: Whether to return neighbor coordinates.
        wp_device: Warp launch device.
        wp_stream: Warp launch stream.

    Returns:
        Tuple ``(indices, points, distances, num_neighbors)`` with batch dim
        intact; ``points``/``distances`` are ``None`` when not requested.
    """
    B, Q, Bq = qprep["B"], qprep["Q"], qprep["Bq"]
    device = index.pc_xyz4.device
    out_idx, out_count, out_dist, out_pts = _alloc_outputs(
        B, Q, max_points, return_dists, return_points, device
    )
    idx_wp, count_wp, dist_wp, pts_wp = _wp_out(
        out_idx, out_count, out_dist, out_pts, Bq, max_points, return_dists, return_points
    )

    num_tasks = tasks["num_tasks"]
    if num_tasks > 0:
        radius2 = float(radius) * float(radius)
        task_handles = [
            wp.from_torch(tasks["task_q_begin"], return_ctype=True),
            wp.from_torch(tasks["task_q_count"], return_ctype=True),
            wp.from_torch(tasks["task_p_begin"], return_ctype=True),
            wp.from_torch(tasks["task_p_count"], return_ctype=True),
        ]
        if variant == "fma":
            # Regular 2D launch: dim=(task, lane); one lane per candidate point.
            wp.launch(
                kernel=mk.tiled_fma_kernel,
                dim=(num_tasks, TILE_P),
                inputs=task_handles
                + [
                    wp.from_torch(qprep["qp_xyz4"], return_ctype=True),
                    wp.from_torch(qprep["q_orig"], return_ctype=True),
                    wp.from_torch(index.pc_xyz4, return_ctype=True),
                    wp.from_torch(index.pc_orig, return_ctype=True),
                    radius2,
                    int(max_points),
                    bool(return_dists),
                    bool(return_points),
                    idx_wp,
                    count_wp,
                    dist_wp,
                    pts_wp,
                ],
                device=wp_device,
                stream=wp_stream,
            )
        else:  # gemm
            lane_iota = torch.arange(TILE_P, dtype=torch.int32, device=device)
            wp.launch_tiled(
                kernel=mk.tiled_gemm_kernel,
                dim=num_tasks,
                inputs=task_handles
                + [
                    wp.from_torch(lane_iota, return_ctype=True),
                    wp.from_torch(qprep["qp_xyz4"], return_ctype=True),
                    wp.from_torch(qprep["qp_norm"], return_ctype=True),
                    wp.from_torch(qprep["q_orig"], return_ctype=True),
                    wp.from_torch(index.pc_xyz4, return_ctype=True),
                    wp.from_torch(index.pc_norm, return_ctype=True),
                    wp.from_torch(index.pc_orig, return_ctype=True),
                    radius2,
                    int(max_points),
                    bool(return_dists),
                    bool(return_points),
                    idx_wp,
                    count_wp,
                    dist_wp,
                    pts_wp,
                ],
                block_dim=TILE_P,
                device=wp_device,
                stream=wp_stream,
            )

    num = out_count.clamp(max=max_points).view(B, Q)
    return out_idx, out_pts, out_dist, num


def radius_search_morton_tiled_fma(index, qprep, tasks, radius, max_points, return_dists, return_points, wp_device, wp_stream):
    """Function B1: tiled direct-FMA search (see :func:`_run_tiled`)."""
    return _run_tiled(
        index, qprep, tasks, "fma", radius, max_points, return_dists, return_points, wp_device, wp_stream
    )


def radius_search_morton_tiled_gemm(index, qprep, tasks, radius, max_points, return_dists, return_points, wp_device, wp_stream):
    """Function B2: tiled GEMM search (see :func:`_run_tiled`)."""
    return _run_tiled(
        index, qprep, tasks, "gemm", radius, max_points, return_dists, return_points, wp_device, wp_stream
    )


def _empty_outputs(B, Q, max_points, return_dists, return_points, was_unbatched, dtype, device):
    """Build a zero/empty 4-tuple matching the op contract (used for empty inputs)."""
    indices = torch.zeros((B, Q, max_points), dtype=torch.int32, device=device)
    num = torch.zeros((B, Q), dtype=torch.int32, device=device)
    pts = (
        torch.zeros((B, Q, max_points, 3), dtype=dtype, device=device)
        if return_points
        else torch.empty((0, max_points, 3), dtype=dtype, device=device)
    )
    dist = (
        torch.zeros((B, Q, max_points), dtype=dtype, device=device)
        if return_dists
        else torch.empty(0, dtype=dtype, device=device)
    )
    if was_unbatched:
        indices = indices.squeeze(0)
        num = num.squeeze(0)
        if return_points:
            pts = pts.squeeze(0)
        if return_dists:
            dist = dist.squeeze(0)
    return indices, pts, dist, num


def radius_search_morton(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    return_dists: bool = False,
    return_points: bool = False,
    variant: str | None = "scalar",
    cell_size_factor: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Morton-grid radius search entry point (CUDA-only, ``max_points`` path).

    Builds the reusable :class:`MortonGridIndex` from ``points``, prepares the
    queries, and dispatches to the requested ``variant``. Returns the same
    4-tuple contract as the ``max_points`` path of ``radius_search_impl`` so the
    custom op's registered autograd/fake paths are unaffected.

    Args:
        points: Reference points, ``(N, 3)`` or ``(B, N, 3)``.
        queries: Query points, ``(M, 3)`` or ``(B, M, 3)``.
        radius: Search radius.
        max_points: Maximum neighbors per query (must not be ``None``).
        return_dists: Whether to return neighbor distances.
        return_points: Whether to return neighbor coordinates.
        variant: One of ``"scalar"`` (A), ``"fma"`` (B1), ``"gemm"`` (B2).
            ``None`` defaults to ``"scalar"``.
        cell_size_factor: ``cell_size / radius`` (1.0 -> 27-cell stencil; use
            0.5 for a denser 125-cell stencil).

    Returns:
        ``(indices, points, distances, num_neighbors)`` mirroring
        ``radius_search_impl``.

    Raises:
        ValueError: If ``max_points`` is ``None``, inputs are not CUDA, or
            ``variant`` is unknown.
    """
    if max_points is None:
        raise ValueError("radius_search_morton requires max_points to be set (not None)")
    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError("radius_search_morton requires CUDA tensors")

    variant = variant or "scalar"
    if variant not in VARIANTS:
        raise ValueError(f"unknown morton variant '{variant}'; expected one of {VARIANTS}")

    points, queries, was_unbatched = validate_inputs(points, queries)
    input_dtype = points.dtype

    if points.dtype != torch.float32:
        points = points.to(torch.float32)
    if queries.dtype != torch.float32:
        queries = queries.to(torch.float32)
    points = points.contiguous()
    queries = queries.contiguous()

    B, N, _ = points.shape
    Q = queries.shape[1]

    if N == 0 or Q == 0:
        return _empty_outputs(
            B, Q, max_points, return_dists, return_points, was_unbatched, input_dtype, points.device
        )

    wp_device, wp_stream = FunctionSpec.warp_launch_context(points)

    with wp.ScopedStream(wp_stream):
        index = build_morton_pc_index(points, radius, cell_size_factor, wp_device, wp_stream)
        qprep = _prepare_queries(index, queries, wp_device, wp_stream)
        if variant == "scalar":
            indices, pts, dist, num = radius_search_morton_scalar_hash(
                index, qprep, radius, max_points, return_dists, return_points, wp_device, wp_stream
            )
        else:
            tasks = _build_tiles_and_tasks(index, qprep, wp_device, wp_stream)
            indices, pts, dist, num = _run_tiled(
                index, qprep, tasks, variant, radius, max_points, return_dists, return_points, wp_device, wp_stream
            )

    if pts is None:
        pts = torch.empty((0, max_points, 3), dtype=torch.float32, device=points.device)
    if dist is None:
        dist = torch.empty(0, dtype=torch.float32, device=points.device)

    if was_unbatched:
        indices = indices.squeeze(0)
        num = num.squeeze(0)
        if return_points:
            pts = pts.squeeze(0)
        if return_dists:
            dist = dist.squeeze(0)

    pts = pts.to(input_dtype)
    dist = dist.to(input_dtype)
    return indices, pts, dist, num
