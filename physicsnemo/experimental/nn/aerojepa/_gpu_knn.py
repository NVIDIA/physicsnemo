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

"""GPU-native kNN utilities (private to ``physicsnemo.experimental.nn.aerojepa``).

Chunked ``torch.cdist`` plus ``topk`` to build homogeneous and bipartite
k-nearest-neighbor graphs and to do inverse-distance interpolation, without
the CPU/GPU round-trip imposed by ``scipy.spatial.cKDTree``. Operations are
pure PyTorch, so the module works on CPU tensors too — it is just slower
there.

This module is intentionally private (leading underscore on the filename):
the AeroJEPA tokenizer and graph builder are the only callers. Public users
should not import from here directly.
"""

from __future__ import annotations

import torch


def _gpu_knn_chunked(
    query: torch.Tensor,
    source: torch.Tensor,
    k: int,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Find the ``k`` nearest source points for each query point.

    Peak memory is bounded at ``chunk_size * n_source * sizeof(float32)``
    per chunk. Inputs in ``bfloat16`` / ``float16`` are promoted to
    ``float32`` for the distance computation and cast back on return.

    Parameters
    ----------
    query : torch.Tensor
        Query positions of shape ``(Nq, D)``.
    source : torch.Tensor
        Source positions of shape ``(Ns, D)``.
    k : int
        Number of nearest neighbors requested per query point. Clamped to
        ``min(k, Ns)``.
    chunk_size : int, optional
        Number of query points processed per chunk. Default 4096.

    Returns
    -------
    distances : torch.Tensor
        Euclidean distances of shape ``(Nq, k_eff)``, dtype matching
        ``query``.
    indices : torch.Tensor
        Source indices of shape ``(Nq, k_eff)``, dtype ``int64``.
    """
    n_query = query.shape[0]
    n_source = source.shape[0]
    k_eff = min(k, n_source)
    device = query.device
    dtype = query.dtype

    if n_query <= chunk_size:
        dist = torch.cdist(
            query.float().unsqueeze(0),
            source.float().unsqueeze(0),
        ).squeeze(0)
        topk_dist, topk_idx = dist.topk(k_eff, dim=1, largest=False)
        return topk_dist.to(dtype), topk_idx

    all_dist = torch.empty((n_query, k_eff), device=device, dtype=dtype)
    all_idx = torch.empty((n_query, k_eff), device=device, dtype=torch.long)
    source_f = source.float()

    for start in range(0, n_query, chunk_size):
        end = min(start + chunk_size, n_query)
        q_chunk = query[start:end].float()
        dist_chunk = torch.cdist(
            q_chunk.unsqueeze(0),
            source_f.unsqueeze(0),
        ).squeeze(0)
        topk_dist, topk_idx = dist_chunk.topk(k_eff, dim=1, largest=False)
        all_dist[start:end] = topk_dist.to(dtype)
        all_idx[start:end] = topk_idx

    return all_dist, all_idx


def gpu_knn_self(
    positions: torch.Tensor,
    k: int,
    *,
    include_self_edges: bool = False,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build a homogeneous k-NN graph from a single point cloud.

    Returns flat edge lists. When ``include_self_edges`` is ``False``, the
    ``k+1`` nearest neighbors are computed and each row's self-edge is
    masked out (by setting that distance to ``inf`` and re-sorting), so
    each node ends up with exactly ``k`` non-self neighbors.

    Parameters
    ----------
    positions : torch.Tensor
        Point positions of shape ``(N, D)``.
    k : int
        Number of neighbors per node. Clamped to ``min(k, N)`` (or
        ``min(k, N-1)`` when self-edges are excluded).
    include_self_edges : bool, optional
        If ``True``, each node may include itself among its neighbors.
        Default ``False``.
    chunk_size : int, optional
        Chunk size for the underlying chunked cdist. Default 4096.

    Returns
    -------
    senders : torch.Tensor
        Source-of-edge indices of shape ``(E,)``, dtype ``int64``.
    receivers : torch.Tensor
        Target-of-edge indices of shape ``(E,)``, dtype ``int64``.
    distances : torch.Tensor
        Edge distances of shape ``(E,)``, dtype matching ``positions``.

    Notes
    -----
    Empty edge tensors are returned when ``N == 0``, or when ``N == 1``
    with ``include_self_edges=False``, or whenever the clamped ``k_eff``
    is zero.
    """
    n = positions.shape[0]
    if n == 0:
        empty = torch.empty((0,), dtype=torch.long, device=positions.device)
        return empty, empty, empty.to(dtype=positions.dtype)
    if n == 1 and not include_self_edges:
        empty = torch.empty((0,), dtype=torch.long, device=positions.device)
        return empty, empty, empty.to(dtype=positions.dtype)

    max_k = n if include_self_edges else max(0, n - 1)
    k_eff = min(k, max_k)
    if k_eff == 0:
        empty = torch.empty((0,), dtype=torch.long, device=positions.device)
        return empty, empty, empty.to(dtype=positions.dtype)

    if include_self_edges:
        dist, idx = _gpu_knn_chunked(positions, positions, k_eff, chunk_size)
    else:
        raw_dist, raw_idx = _gpu_knn_chunked(
            positions, positions, k_eff + 1, chunk_size
        )
        self_id = torch.arange(n, device=positions.device).unsqueeze(1)
        is_self = raw_idx == self_id
        raw_dist_masked = raw_dist.clone()
        raw_dist_masked[is_self] = float("inf")
        sorted_dist, sort_order = raw_dist_masked.sort(dim=1)
        sorted_idx = raw_idx.gather(1, sort_order)
        dist = sorted_dist[:, :k_eff]
        idx = sorted_idx[:, :k_eff]

    receivers = (
        torch.arange(n, device=positions.device)
        .unsqueeze(1)
        .expand(-1, k_eff)
        .reshape(-1)
    )
    senders = idx.reshape(-1)
    distances = dist.reshape(-1)
    return senders.long(), receivers.long(), distances


def gpu_knn_bipartite(
    sender_positions: torch.Tensor,
    receiver_positions: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Build a bipartite k-NN graph: each receiver finds its ``k`` nearest senders.

    Parameters
    ----------
    sender_positions : torch.Tensor
        Sender positions of shape ``(Ns, D)``.
    receiver_positions : torch.Tensor
        Receiver positions of shape ``(Nr, D)``.
    k : int
        Number of nearest senders per receiver. Clamped to
        ``min(k, Ns)``.
    chunk_size : int, optional
        Chunk size for the underlying chunked cdist. Default 4096.

    Returns
    -------
    senders : torch.Tensor
        Sender-side edge indices of shape ``(E,)``, dtype ``int64``.
    receivers : torch.Tensor
        Receiver-side edge indices of shape ``(E,)``, dtype ``int64``.
    distances : torch.Tensor
        Edge distances of shape ``(E,)``, dtype matching the inputs.

    Raises
    ------
    ValueError
        If ``sender_positions`` has zero rows.
    """
    n_sender = sender_positions.shape[0]
    n_receiver = receiver_positions.shape[0]
    if n_receiver == 0:
        empty = torch.empty(
            (0,), dtype=torch.long, device=sender_positions.device
        )
        return empty, empty, empty.to(dtype=sender_positions.dtype)
    if n_sender == 0:
        raise ValueError("sender_positions must have at least one point")

    k_eff = min(k, n_sender)
    dist, idx = _gpu_knn_chunked(
        receiver_positions, sender_positions, k_eff, chunk_size
    )

    receivers = (
        torch.arange(n_receiver, device=receiver_positions.device)
        .unsqueeze(1)
        .expand(-1, k_eff)
        .reshape(-1)
    )
    senders = idx.reshape(-1)
    distances = dist.reshape(-1)
    return senders.long(), receivers.long(), distances


def gpu_knn_interpolate(
    src_pos: torch.Tensor,
    src_feat: torch.Tensor,
    query_pos: torch.Tensor,
    k: int,
    *,
    chunk_size: int = 4096,
    eps: float = 1e-9,
) -> torch.Tensor:
    r"""Inverse-distance-weighted k-NN interpolation.

    For each query point, gather the ``k`` nearest source features and
    return a weighted sum where the weight of each neighbor is
    ``1 / (distance + eps)``, normalized so the weights sum to one.

    Parameters
    ----------
    src_pos : torch.Tensor
        Source positions of shape ``(Ns, D)``.
    src_feat : torch.Tensor
        Source features of shape ``(Ns, F)``.
    query_pos : torch.Tensor
        Query positions of shape ``(Nq, D)``.
    k : int
        Number of source neighbors used per query. Clamped to
        ``min(max(1, k), Ns)``.
    chunk_size : int, optional
        Chunk size for the gather + weighted-sum pass; if zero or
        greater than ``Nq``, the whole thing runs in one shot. Default
        4096.
    eps : float, optional
        Numerical floor added to distances before inverting and to the
        per-row weight sum before dividing. Default ``1e-9``.

    Returns
    -------
    interpolated : torch.Tensor
        Interpolated features of shape ``(Nq, F)``.

    Raises
    ------
    ValueError
        If ``src_pos`` has zero rows.
    """
    n_src = src_pos.shape[0]
    if n_src == 0:
        raise ValueError("src_pos must have at least one point")
    k_eff = min(max(1, k), n_src)

    n_query = query_pos.shape[0]
    if n_query == 0:
        return src_feat.new_empty((0, src_feat.shape[-1]))

    dist, idx = _gpu_knn_chunked(query_pos, src_pos, k_eff, chunk_size)

    w = 1.0 / (dist + eps)
    w = w / w.sum(dim=1, keepdim=True).clamp_min(eps)

    if n_query <= chunk_size or chunk_size <= 0:
        gathered = src_feat.index_select(0, idx.reshape(-1)).reshape(
            n_query, k_eff, -1
        )
        return (gathered * w.unsqueeze(-1)).sum(dim=1)

    out_chunks = []
    for s in range(0, n_query, chunk_size):
        e = min(s + chunk_size, n_query)
        idx_c = idx[s:e]
        w_c = w[s:e]
        gathered = src_feat.index_select(0, idx_c.reshape(-1)).reshape(
            e - s, k_eff, -1
        )
        out_chunks.append((gathered * w_c.unsqueeze(-1)).sum(dim=1))
    return torch.cat(out_chunks, dim=0)
