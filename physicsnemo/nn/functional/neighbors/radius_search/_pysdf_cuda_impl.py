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

"""Torch <-> CUDA glue for the ``pysdf_cuda`` radius-search backend.

This backend reuses NVIDIA pysdf's software-QBVH point-range-query (the no-OptiX
BVH from ``minigql`` / ``owl``, vendored under ``_pysdf_cuda_ext/third_party``).
The CUDA sources are JIT-compiled on first use via
:func:`torch.utils.cpp_extension.load` and cached in ``TORCH_EXTENSIONS_DIR``;
nothing is compiled unless the backend is actually selected, so a CUDA toolkit
(``nvcc``) is only required when ``PHYSICSNEMO_RADIUS_SEARCH_MORTON=pysdf_cuda``.

:func:`radius_search_pysdf_cuda` is the single entry point used by the dispatch
in :mod:`._warp_impl`. It only supports the deterministic (``max_points``) path
on CUDA tensors and returns the same 4-tuple contract as the other backends.
"""

from __future__ import annotations

import functools
import pathlib

import torch

from .utils import validate_inputs

_EXT_NAME = "physicsnemo_pysdf_radius_search"
_EXT_DIR = pathlib.Path(__file__).parent / "_pysdf_cuda_ext"


@functools.lru_cache(maxsize=1)
def _load_ext():
    """JIT-compile (once) and return the vendored pysdf CUDA extension module.

    Returns:
        The loaded extension module exposing ``radius_search_pysdf_cuda_single``.

    Raises:
        RuntimeError: If the vendored sources are missing or compilation fails
            (e.g. no CUDA toolkit / ``nvcc`` available).
    """
    from torch.utils.cpp_extension import load

    third_party = _EXT_DIR / "third_party"
    sources = [
        str(_EXT_DIR / "radius_search_ext.cu"),
        str(third_party / "gequel" / "bvhLib" / "spatialMedianBuilder.cu"),
    ]
    for src in sources:
        if not pathlib.Path(src).is_file():
            raise RuntimeError(
                f"pysdf_cuda backend source not found: {src}. The vendored "
                "sources under _pysdf_cuda_ext must ship with the package."
            )
    return load(
        name=_EXT_NAME,
        sources=sources,
        extra_include_paths=[str(third_party)],
        extra_cuda_cflags=[
            "--extended-lambda",
            "--expt-relaxed-constexpr",
            "-DBVH_IS_3D",
            "-lineinfo",
        ],
        verbose=False,
    )


def _empty_outputs(
    B: int,
    Q: int,
    max_points: int,
    return_dists: bool,
    return_points: bool,
    was_unbatched: bool,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build zero-sized/zero-filled outputs for empty point clouds or queries."""
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


def radius_search_pysdf_cuda(
    points: torch.Tensor,
    queries: torch.Tensor,
    radius: float,
    max_points: int,
    return_dists: bool = False,
    return_points: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """pysdf software-QBVH radius search entry point (CUDA-only, ``max_points``).

    Builds a QBVH over ``points`` and, for each query, traverses it once keeping
    the ``max_points`` nearest in-radius neighbors. Returns the same 4-tuple
    contract as the ``max_points`` path of ``radius_search_impl``.

    Args:
        points: Reference points, ``(N, 3)`` or ``(B, N, 3)``.
        queries: Query points, ``(M, 3)`` or ``(B, M, 3)``.
        radius: Search radius.
        max_points: Maximum neighbors per query (must not be ``None``).
        return_dists: Whether to return neighbor distances.
        return_points: Whether to return neighbor coordinates.

    Returns:
        ``(indices, points, distances, num_neighbors)`` mirroring
        ``radius_search_impl``. ``points``/``distances`` are empty tensors when
        not requested; ``num_neighbors`` is capped at ``max_points``.

    Raises:
        ValueError: If ``max_points`` is ``None`` or inputs are not CUDA tensors.
    """
    if max_points is None:
        raise ValueError("radius_search_pysdf_cuda requires max_points (not None)")
    if points.device != queries.device:
        raise ValueError("points and queries must be on the same device")
    if points.device.type != "cuda":
        raise ValueError("radius_search_pysdf_cuda requires CUDA tensors")

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
            B, Q, max_points, return_dists, return_points, was_unbatched,
            input_dtype, points.device,
        )

    ext = _load_ext()

    # One (points, queries) pair at a time; the extension owns the QBVH build
    # and the per-query range query for that batch element.
    idx_list, pts_list, dist_list, num_list = [], [], [], []
    for b in range(B):
        out_idx, out_pts, out_dist, out_count = ext.radius_search_pysdf_cuda_single(
            points[b],
            queries[b],
            float(radius),
            int(max_points),
            bool(return_dists),
            bool(return_points),
        )
        idx_list.append(out_idx)
        pts_list.append(out_pts)
        dist_list.append(out_dist)
        num_list.append(out_count)

    indices = torch.stack(idx_list, dim=0)  # (B, Q, max_points)
    num = torch.stack(num_list, dim=0)  # (B, Q)
    if return_points:
        pts = torch.stack(pts_list, dim=0)  # (B, Q, max_points, 3)
    else:
        pts = torch.empty((0, max_points, 3), dtype=torch.float32, device=points.device)
    if return_dists:
        dist = torch.stack(dist_list, dim=0)  # (B, Q, max_points)
    else:
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
