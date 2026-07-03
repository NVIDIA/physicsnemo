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

"""Generic batching, masking and k-NN utilities for point-cloud transformers.

Helpers for converting between padded and flat point/token representations,
computing per-batch coordinate offsets, gathering rows, building dilated
k-NN indices, and reducing over a token axis with optional masking. None of
these depend on AeroJEPA-specific dataclasses; they are reusable building
blocks for any point-cloud transformer layer.
"""

from __future__ import annotations

import torch

from physicsnemo.nn.functional.neighbors.knn import knn


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    r"""Gather rows of ``x`` by an arbitrary-shape index tensor.

    Parameters
    ----------
    x : torch.Tensor
        Source tensor of shape ``(N, F)``.
    idx : torch.Tensor
        Index tensor of any shape ``(*S,)`` holding integer indices into the
        first axis of ``x``.

    Returns
    -------
    torch.Tensor
        Tensor of shape ``(*S, F)`` whose entries are
        ``x[idx[*s]]``.
    """
    flat = idx.reshape(-1)
    gathered = x.index_select(0, flat)
    return gathered.reshape(*idx.shape, x.shape[-1])


def counts_to_mask(counts: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    r"""Convert per-row valid-element counts to a 2-D boolean mask.

    Parameters
    ----------
    counts : torch.Tensor
        Rank-1 tensor of shape ``(B,)`` with the number of valid entries
        per row.
    max_len : int, optional
        Number of columns in the output mask. If ``None``, uses
        ``counts.max()`` (or 0 when ``counts`` is empty).

    Returns
    -------
    torch.Tensor
        Boolean tensor of shape ``(B, max_len)`` with ``True`` for the
        first ``counts[i]`` entries of each row.

    Raises
    ------
    ValueError
        If ``counts`` is not rank 1.
    """
    if counts.ndim != 1:
        raise ValueError(
            f"counts_to_mask expects rank-1 counts, got {tuple(counts.shape)}"
        )
    if max_len is None:
        max_len = int(counts.max().item()) if int(counts.numel()) > 0 else 0
    arange = torch.arange(int(max_len), device=counts.device, dtype=counts.dtype)
    return arange.unsqueeze(0) < counts.unsqueeze(1)


def flatten_padded_batch(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Select valid entries of a padded batched tensor using a boolean mask.

    Parameters
    ----------
    x : torch.Tensor
        Padded tensor of shape ``(B, N, *F)``.
    mask : torch.Tensor
        Boolean tensor of shape ``(B, N)`` whose ``True`` entries identify
        valid positions.

    Returns
    -------
    torch.Tensor
        Flat tensor of shape ``(M, *F)`` containing the ``mask.sum()`` valid
        entries concatenated row-major.

    Raises
    ------
    ValueError
        If ``mask`` is not rank 2 or its shape doesn't match
        ``x.shape[:2]``.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"flatten_padded_batch expects rank-2 mask, got {tuple(mask.shape)}"
        )
    if x.shape[:2] != mask.shape:
        raise ValueError(
            "flatten_padded_batch expects x.shape[:2] == mask.shape, "
            f"got {tuple(x.shape[:2])} vs {tuple(mask.shape)}"
        )
    return x[mask]


def unflatten_to_padded(flat: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    r"""Scatter a flat tensor back into a padded batched layout.

    Inverse of :func:`flatten_padded_batch`. Positions where ``mask`` is
    ``False`` are filled with zeros.

    Parameters
    ----------
    flat : torch.Tensor
        Flat tensor of shape ``(M, F)`` whose row count must equal
        ``mask.sum()``.
    mask : torch.Tensor
        Boolean tensor of shape ``(B, N)`` identifying valid target
        positions.

    Returns
    -------
    torch.Tensor
        Padded tensor of shape ``(B, N, F)``.

    Raises
    ------
    ValueError
        If ranks are wrong or row counts do not match.
    """
    if mask.ndim != 2:
        raise ValueError(
            f"unflatten_to_padded expects rank-2 mask, got {tuple(mask.shape)}"
        )
    if flat.ndim != 2:
        raise ValueError(
            f"unflatten_to_padded expects rank-2 flat tensor, got {tuple(flat.shape)}"
        )
    if int(mask.sum().item()) != int(flat.shape[0]):
        raise ValueError(
            "flat row count must match mask true count, "
            f"got {flat.shape[0]} vs {int(mask.sum().item())}"
        )
    out = flat.new_zeros(
        (int(mask.shape[0]), int(mask.shape[1]), int(flat.shape[-1]))
    )
    out[mask] = flat
    return out


def compute_batch_offset_step(
    coords: torch.Tensor, mask: torch.Tensor | None = None
) -> float:
    r"""Compute a safe per-batch offset step from coordinate extents.

    Returns ``max(4 * (span + 1), 1)`` where ``span`` is the largest
    coordinate range across any axis of the valid entries. Used by
    :func:`flatten_batched_coords` to push different batch items far enough
    apart along the first coordinate axis that they never collide.

    Parameters
    ----------
    coords : torch.Tensor
        Coordinate tensor of shape ``(N, D)`` or ``(B, N, D)``.
    mask : torch.Tensor, optional
        Validity mask of shape ``(B, N)``. Only meaningful when ``coords``
        is rank 3.

    Returns
    -------
    float
        Offset step. Falls back to ``1.0`` when there are no valid points.

    Raises
    ------
    ValueError
        If ``coords`` is not rank 2 or 3, if ``mask`` is supplied for a
        rank-2 ``coords``, or if ``mask`` shape does not match
        ``coords.shape[:2]``.
    """
    if coords.ndim not in {2, 3}:
        raise ValueError(
            f"compute_batch_offset_step expects rank-2/3 coords, got {tuple(coords.shape)}"
        )
    if mask is not None and coords.ndim != 3:
        raise ValueError("mask is only supported for rank-3 coords.")

    if coords.ndim == 2:
        valid = coords
    else:
        if mask is None:
            valid = coords.reshape(-1, int(coords.shape[-1]))
        else:
            if mask.shape != coords.shape[:2]:
                raise ValueError(
                    "mask must match coords batch/token dims, "
                    f"got {tuple(mask.shape)} vs {tuple(coords.shape[:2])}"
                )
            valid = coords[mask]
    if int(valid.shape[0]) == 0:
        return 1.0
    span = float(
        (valid.max(dim=0).values - valid.min(dim=0).values).abs().max().item()
    )
    return max(4.0 * (span + 1.0), 1.0)


def flatten_batched_coords(
    coords: torch.Tensor,
    mask: torch.Tensor,
    *,
    offset_step: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    r"""Flatten a padded batched coordinate tensor and apply per-batch offset.

    The flat tensor concatenates valid points from each batch item, and
    ``flat_offset_coords`` shifts the first coordinate axis by
    ``batch_id * offset_step`` so that points from different batches occupy
    disjoint regions of space — useful when a downstream k-NN should not mix
    across batches.

    Parameters
    ----------
    coords : torch.Tensor
        Batched coordinates of shape ``(B, N, D)``.
    mask : torch.Tensor
        Boolean mask of shape ``(B, N)`` for valid points.
    offset_step : float
        Per-batch shift along the first coordinate axis. Typically obtained
        from :func:`compute_batch_offset_step`.

    Returns
    -------
    flat_coords : torch.Tensor
        Concatenated valid coordinates of shape ``(M, D)``, unshifted.
    flat_offset_coords : torch.Tensor
        Same as ``flat_coords`` but with the first axis offset per batch.
    batch_ids : torch.Tensor
        ``int64`` batch index for each valid point, shape ``(M,)``.

    Raises
    ------
    ValueError
        If ranks or shapes do not match the expected layout.
    """
    if coords.ndim != 3 or mask.ndim != 2:
        raise ValueError(
            "flatten_batched_coords expects rank-3 coords and rank-2 mask."
        )
    if coords.shape[:2] != mask.shape:
        raise ValueError(
            "flatten_batched_coords expects coords.shape[:2] == mask.shape, "
            f"got {tuple(coords.shape[:2])} vs {tuple(mask.shape)}"
        )
    flat_coords = coords[mask]
    batch_ids = (
        torch.arange(int(coords.shape[0]), device=coords.device, dtype=torch.long)
        .unsqueeze(1)
        .expand_as(mask)[mask]
    )
    flat_offset_coords = flat_coords.clone()
    flat_offset_coords[:, 0] = flat_offset_coords[:, 0] + batch_ids.to(
        dtype=flat_offset_coords.dtype
    ) * float(offset_step)
    return flat_coords, flat_offset_coords, batch_ids


def chunked_knn_indices(
    *,
    query_coords: torch.Tensor,
    key_coords: torch.Tensor,
    k: int,
    chunk_size: int,
    dilation: int = 1,
) -> torch.Tensor:
    r"""Build a k-NN index from queries to keys with optional dilation.

    Thin wrapper over :func:`physicsnemo.nn.functional.neighbors.knn.knn`:
    runs the canonical kNN search once at ``k * dilation`` neighbors,
    then applies the dilation stride and keeps the first ``k`` indices
    per query.

    Parameters
    ----------
    query_coords : torch.Tensor
        Query positions of shape ``(Nq, D)``.
    key_coords : torch.Tensor
        Key positions of shape ``(Nk, D)``.
    k : int
        Number of nearest neighbors retained per query (post-dilation).
    chunk_size : int
        Accepted for call-site signature compatibility; the canonical kNN
        handles chunking internally and this value is unused.
    dilation : int, optional
        Stride applied to the top-``k * dilation`` indices before keeping
        the first ``k``. Lets blocks subsample at coarser scales without
        rerunning the search. Default 1.

    Returns
    -------
    torch.Tensor
        Index tensor of shape ``(Nq, k_eff)`` with dtype ``int64``.

    Raises
    ------
    ValueError
        If ranks or coordinate dimensions disagree, if either coord set is
        empty, or if any of ``k``, ``chunk_size``, ``dilation`` are
        non-positive.
    """
    if query_coords.ndim != 2 or key_coords.ndim != 2:
        raise ValueError(
            "chunked_knn_indices expects rank-2 query_coords and key_coords."
        )
    if query_coords.shape[-1] != key_coords.shape[-1]:
        raise ValueError(
            "query_coords and key_coords must have the same coordinate dimension."
        )
    if int(query_coords.shape[0]) == 0 or int(key_coords.shape[0]) == 0:
        raise ValueError(
            "query_coords and key_coords must both contain at least one point."
        )
    if k <= 0 or chunk_size <= 0 or dilation <= 0:
        raise ValueError("k, chunk_size, and dilation must be positive.")

    k_request = min(int(k) * int(dilation), int(key_coords.shape[0]))
    idx, _ = knn(points=key_coords, queries=query_coords, k=k_request)
    idx = idx.to(dtype=torch.long)
    if dilation > 1:
        idx = idx[:, :: int(dilation)]
    out_k = min(max(1, int(k)), idx.shape[1])
    return idx[:, :out_k]


def masked_mean(
    features: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    r"""Compute a mean of ``features`` along the token axis, optionally masked.

    For rank-2 ``features`` of shape ``(N, F)``, averages along ``N``; for
    rank-3 ``(B, N, F)``, averages along ``N`` per batch item. When ``mask``
    is provided, only entries where ``mask`` is ``True`` contribute, with
    the denominator floored at 1 to handle empty rows.

    Parameters
    ----------
    features : torch.Tensor
        Features of shape ``(N, F)`` or ``(B, N, F)``.
    mask : torch.Tensor, optional
        Boolean mask of shape ``(N,)`` or ``(B, N)``. ``None`` means no
        masking — equivalent to a plain mean.

    Returns
    -------
    torch.Tensor
        Mean of shape ``(1, F)`` (rank-2 input) or ``(B, 1, F)`` (rank-3
        input).
    """
    if mask is None:
        return (
            features.mean(dim=0, keepdim=True)
            if features.ndim == 2
            else features.mean(dim=1, keepdim=True)
        )
    weights = mask.to(dtype=features.dtype).unsqueeze(-1)
    denom = weights.sum(
        dim=-2 if features.ndim == 3 else 0, keepdim=True
    ).clamp_min(1.0)
    summed = (features * weights).sum(
        dim=-2 if features.ndim == 3 else 0, keepdim=True
    )
    return summed / denom
