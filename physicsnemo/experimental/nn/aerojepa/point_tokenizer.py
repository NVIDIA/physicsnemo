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

"""Point-cloud tokenizer for AeroJEPA.

``PointCloudTokenizer`` reduces a raw point set to a bounded token budget
before the attention stack runs. Several tokenization strategies are
supported, from simple identity / random / farthest-point sampling to
voxel-pooled FPS and prototype-anchored clustering. When a clustering
strategy is used, each token's features are the mean of its k-nearest
source points.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

from .token_utils import chunked_knn_indices


def _farthest_point_sampling(
    points: torch.Tensor,
    *,
    num_samples: int,
    random_start: bool,
) -> torch.Tensor:
    n = int(points.shape[0])
    if num_samples >= n:
        return torch.arange(n, device=points.device, dtype=torch.long)
    if num_samples <= 0:
        raise ValueError("num_samples must be > 0")

    selected = torch.empty((num_samples,), device=points.device, dtype=torch.long)
    if random_start:
        current = torch.randint(
            0, n, (1,), device=points.device, dtype=torch.long
        ).item()
    else:
        current = 0
    selected[0] = current
    min_dist_sq = torch.full(
        (n,), float("inf"), device=points.device, dtype=points.dtype
    )

    for i in range(1, num_samples):
        ref = points[current : current + 1]
        dist_sq = torch.sum((points - ref) ** 2, dim=-1)
        min_dist_sq = torch.minimum(min_dist_sq, dist_sq)
        current = int(torch.argmax(min_dist_sq).item())
        selected[i] = current
    return selected


def _normalize_voxel_size(
    voxel_size: Sequence[float] | float,
    *,
    coord_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(voxel_size, (int, float)):
        values = [float(voxel_size)] * int(coord_dim)
    else:
        values = [float(v) for v in voxel_size]
    if len(values) != int(coord_dim):
        raise ValueError(
            f"voxel_size must have length {coord_dim}, got {len(values)}"
        )
    out = torch.tensor(values, device=device, dtype=dtype)
    if torch.any(out <= 0):
        raise ValueError("voxel_size values must be > 0.")
    return out


def _voxel_pool_positions(
    *,
    point_positions: torch.Tensor,
    voxel_size: Sequence[float] | float,
) -> torch.Tensor:
    if point_positions.ndim != 2:
        raise ValueError("_voxel_pool_positions expects rank-2 point_positions.")
    voxel_size_t = _normalize_voxel_size(
        voxel_size,
        coord_dim=int(point_positions.shape[1]),
        device=point_positions.device,
        dtype=point_positions.dtype,
    )
    origin = point_positions.min(dim=0).values
    voxel_coords = torch.floor((point_positions - origin) / voxel_size_t).to(
        torch.long
    )
    _, inverse = torch.unique(voxel_coords, dim=0, return_inverse=True)
    num_voxels = int(inverse.max().item()) + 1

    pooled_positions = torch.zeros(
        (num_voxels, int(point_positions.shape[1])),
        device=point_positions.device,
        dtype=point_positions.dtype,
    )
    pooled_positions.index_add_(0, inverse, point_positions)
    counts = torch.zeros(
        (num_voxels, 1), device=point_positions.device, dtype=point_positions.dtype
    )
    counts.index_add_(
        0,
        inverse,
        torch.ones(
            (int(point_positions.shape[0]), 1),
            device=point_positions.device,
            dtype=point_positions.dtype,
        ),
    )
    return pooled_positions / counts.clamp_min(1.0)


class PointCloudTokenizer(nn.Module):
    r"""Reduce a raw point set to a bounded token budget before attention.

    Supports seven tokenization strategies:

    - ``identity``: no reduction; pass features and positions through.
    - ``random``: random subset (uniform without replacement during
      training; deterministic ``linspace`` at eval when
      ``deterministic_eval`` is set).
    - ``fps``: farthest-point sampling.
    - ``random_cluster``: random subset of token centers, then each token
      pools the mean of its ``cluster_size`` nearest source points.
    - ``fps_cluster``: same as ``random_cluster`` but FPS-sampled centers.
    - ``voxel_fps_cluster``: voxel-pool the input to candidate positions
      first, then FPS over those, then cluster pool.
    - ``data_prototype_cluster``: token centers are fixed
      ``prototype_coords``; each token pools the ``prototype_knn_k``
      (or ``cluster_size``) nearest source points.

    Parameters
    ----------
    max_point_tokens : int, optional
        Maximum number of tokens after reduction. ``None`` (or any value
        when ``strategy='identity'``) disables the reduction.
    strategy : str, optional
        Tokenization strategy; one of the seven names listed above.
        Default ``"identity"``.
    deterministic_eval : bool, optional
        When ``True`` and the module is in eval mode, the ``random`` family
        of strategies uses a deterministic stride instead of a random
        shuffle. Default ``True``.
    cluster_size : int, optional
        Number of source points each token aggregates over. Required for
        the ``*_cluster`` strategies. Falls back to ``max_point_tokens``
        when ``None`` for the non-prototype strategies.
    knn_chunk_size : int, optional
        Chunk size forwarded to the k-NN lookups. Default 128.
    voxel_size : float or Sequence[float], optional
        Voxel edge length for the ``voxel_fps_cluster`` strategy. A single
        float is broadcast across coordinate axes; a sequence must match
        the coordinate dimension.
    prototype_coords : torch.Tensor, optional
        Fixed token centers of shape ``(P, 3)`` for the
        ``data_prototype_cluster`` strategy. Registered as a
        non-persistent buffer (not saved in checkpoints).
    prototype_knn_k : int, optional
        Override for the cluster size used by ``data_prototype_cluster``.
        Falls back to ``cluster_size`` (and finally 16) when ``None``.

    Shape
    -----
    - Single-cloud ``forward``: ``point_positions`` and ``point_features``
      are rank-2 of shapes ``(N, D)`` and ``(N, F)``; returns
      ``(token_positions, token_features)`` each of shape ``(T, *)``.
    - Batched ``forward_batched``: rank-3 inputs of shapes ``(B, N, D)``
      and ``(B, N, F)`` plus an optional ``point_counts`` of shape
      ``(B,)``; returns padded outputs of shape ``(B, T_max, *)`` plus a
      mask ``(B, T_max)`` and per-batch token counts ``(B,)``.

    Raises
    ------
    ValueError
        If ``strategy`` is unknown, if a strategy's required parameter is
        missing (e.g. ``voxel_size`` for ``voxel_fps_cluster``,
        ``prototype_coords`` for ``data_prototype_cluster``), or if any of
        the size parameters is non-positive.
    """

    def __init__(
        self,
        *,
        max_point_tokens: int | None = None,
        strategy: str = "identity",
        deterministic_eval: bool = True,
        cluster_size: int | None = None,
        knn_chunk_size: int = 128,
        voxel_size: Sequence[float] | float | None = None,
        prototype_coords: torch.Tensor | None = None,
        prototype_knn_k: int | None = None,
    ):
        super().__init__()
        self.max_point_tokens = (
            None if max_point_tokens is None else int(max_point_tokens)
        )
        if self.max_point_tokens is not None and self.max_point_tokens <= 0:
            raise ValueError("max_point_tokens must be > 0 when provided.")
        self.strategy = str(strategy).lower()
        self.deterministic_eval = bool(deterministic_eval)
        self.cluster_size = None if cluster_size is None else int(cluster_size)
        if self.cluster_size is not None and self.cluster_size <= 0:
            raise ValueError("cluster_size must be > 0 when provided.")
        self.knn_chunk_size = int(knn_chunk_size)
        if self.knn_chunk_size <= 0:
            raise ValueError("knn_chunk_size must be > 0.")
        self.voxel_size = voxel_size
        if prototype_coords is None:
            prototype_coords_t = torch.empty((0, 3), dtype=torch.float32)
        else:
            prototype_coords_t = prototype_coords.detach().float().clone()
            if (
                prototype_coords_t.ndim != 2
                or int(prototype_coords_t.shape[-1]) != 3
            ):
                raise ValueError("prototype_coords must have shape [P, 3].")
        self.register_buffer(
            "prototype_coords", prototype_coords_t, persistent=False
        )
        self.prototype_knn_k = (
            None if prototype_knn_k is None else int(prototype_knn_k)
        )
        if self.prototype_knn_k is not None and self.prototype_knn_k <= 0:
            raise ValueError("prototype_knn_k must be > 0 when provided.")
        if self.strategy not in {
            "identity",
            "random",
            "fps",
            "random_cluster",
            "fps_cluster",
            "voxel_fps_cluster",
            "data_prototype_cluster",
        }:
            raise ValueError(
                "PointCloudTokenizer.strategy must be one of: "
                "'identity', 'random', 'fps', 'random_cluster', 'fps_cluster', "
                "'voxel_fps_cluster', 'data_prototype_cluster'"
            )
        if (
            self.strategy == "data_prototype_cluster"
            and int(self.prototype_coords.shape[0]) <= 0
        ):
            raise ValueError(
                "prototype_coords must be provided for "
                "tokenizer_strategy='data_prototype_cluster'."
            )

    @property
    def uses_cluster_pooling(self) -> bool:
        r"""Whether the configured strategy aggregates over source-point neighborhoods."""
        return self.strategy in {
            "random_cluster",
            "fps_cluster",
            "voxel_fps_cluster",
            "data_prototype_cluster",
        }

    @property
    def requires_per_sample_loop(self) -> bool:
        r"""Whether the configured strategy must run one sample at a time."""
        return self.strategy == "data_prototype_cluster"

    def _select_token_indices(
        self,
        *,
        point_positions: torch.Tensor,
        max_tokens: int,
    ) -> torch.Tensor:
        n_points = int(point_positions.shape[0])
        if self.strategy in {"random", "random_cluster"}:
            if self.training:
                return torch.randperm(n_points, device=point_positions.device)[
                    :max_tokens
                ]
            if self.deterministic_eval:
                return torch.linspace(
                    0, n_points - 1, steps=max_tokens, device=point_positions.device
                ).long()
            return torch.randperm(n_points, device=point_positions.device)[
                :max_tokens
            ]
        return _farthest_point_sampling(
            point_positions,
            num_samples=max_tokens,
            random_start=bool(self.training or not self.deterministic_eval),
        )

    def _select_token_positions(
        self,
        *,
        point_positions: torch.Tensor,
        max_tokens: int,
    ) -> torch.Tensor:
        if self.strategy != "voxel_fps_cluster":
            idx = self._select_token_indices(
                point_positions=point_positions,
                max_tokens=max_tokens,
            )
            return point_positions[idx]

        if self.voxel_size is None:
            raise ValueError(
                "voxel_size must be provided for "
                "tokenizer_strategy='voxel_fps_cluster'."
            )
        candidate_positions = _voxel_pool_positions(
            point_positions=point_positions,
            voxel_size=self.voxel_size,
        )
        if int(candidate_positions.shape[0]) <= max_tokens:
            return candidate_positions
        idx = _farthest_point_sampling(
            candidate_positions,
            num_samples=max_tokens,
            random_start=bool(self.training or not self.deterministic_eval),
        )
        return candidate_positions[idx]

    def tokenize_with_clusters(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        r"""Reduce a single point set to tokens and expose per-token cluster indices.

        Like :meth:`forward` but also returns the cluster-index tensor used
        for pooling so downstream code (e.g. positional bias computation)
        can reuse it without redoing the k-NN.

        Parameters
        ----------
        point_positions : torch.Tensor
            Rank-2 positions of shape ``(N, D)``.
        point_features : torch.Tensor
            Rank-2 features of shape ``(N, F)``.

        Returns
        -------
        token_positions : torch.Tensor
            Shape ``(T, D)``.
        token_features : torch.Tensor
            Shape ``(T, F)``.
        cluster_idx : torch.Tensor or None
            Shape ``(T, K)`` ``int64`` indices into the source points,
            one row per token, when a cluster-pooling strategy is active;
            ``None`` for ``identity`` / ``random`` / ``fps``.

        Raises
        ------
        ValueError
            If the input ranks or row counts disagree.
        """
        if point_positions.ndim != 2 or point_features.ndim != 2:
            raise ValueError(
                "PointCloudTokenizer expects rank-2 point_positions and point_features."
            )
        if point_positions.shape[0] != point_features.shape[0]:
            raise ValueError(
                "point_positions and point_features must agree on point count, "
                f"got {point_positions.shape[0]} and {point_features.shape[0]}"
            )

        n_points = int(point_positions.shape[0])
        if self.strategy == "data_prototype_cluster":
            token_positions = self.prototype_coords.to(
                device=point_positions.device, dtype=point_positions.dtype
            )
            k = (
                self.prototype_knn_k
                if self.prototype_knn_k is not None
                else self.cluster_size
            )
            k = max(1, min(int(k if k is not None else 16), n_points))
            cluster_idx = chunked_knn_indices(
                query_coords=token_positions,
                key_coords=point_positions,
                k=k,
                chunk_size=self.knn_chunk_size,
            )
            cluster_features = point_features[cluster_idx]
            token_features = cluster_features.mean(dim=1)
            return token_positions, token_features, cluster_idx
        if (
            self.max_point_tokens is None
            or self.strategy == "identity"
            or n_points <= self.max_point_tokens
        ):
            return point_positions, point_features, None

        max_tokens = int(self.max_point_tokens)
        if self.strategy in {"random", "fps"}:
            idx = self._select_token_indices(
                point_positions=point_positions,
                max_tokens=max_tokens,
            )
            return point_positions[idx], point_features[idx], None

        token_positions = self._select_token_positions(
            point_positions=point_positions,
            max_tokens=max_tokens,
        )

        cluster_size = (
            max_tokens if self.cluster_size is None else int(self.cluster_size)
        )
        cluster_idx = chunked_knn_indices(
            query_coords=token_positions,
            key_coords=point_positions,
            k=cluster_size,
            chunk_size=self.knn_chunk_size,
        )
        cluster_features = point_features[cluster_idx]
        token_features = cluster_features.mean(dim=1)
        return token_positions, token_features, cluster_idx

    def forward(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_positions, token_features, _ = self.tokenize_with_clusters(
            point_positions=point_positions,
            point_features=point_features,
        )
        return token_positions, token_features

    def forward_batched(
        self,
        *,
        point_positions: torch.Tensor,
        point_features: torch.Tensor,
        point_counts: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Tokenize a padded batched point cloud one sample at a time.

        Some strategies (notably ``data_prototype_cluster`` and
        ``voxel_fps_cluster``) cannot be vectorized cleanly across batch
        items, so this method runs :meth:`forward` per sample and packs
        the variable-length outputs into a padded tensor.

        Parameters
        ----------
        point_positions : torch.Tensor
            Padded positions of shape ``(B, N, D)``.
        point_features : torch.Tensor
            Padded features of shape ``(B, N, F)``.
        point_counts : torch.Tensor, optional
            ``int64`` valid-point counts of shape ``(B,)``. When ``None``,
            assumes all ``N`` slots are valid for each batch item.

        Returns
        -------
        padded_positions : torch.Tensor
            Shape ``(B, T_max, D)``.
        padded_features : torch.Tensor
            Shape ``(B, T_max, F)``.
        token_mask : torch.Tensor
            Boolean mask of shape ``(B, T_max)``; ``True`` for real
            tokens.
        token_counts : torch.Tensor
            ``int64`` tensor of shape ``(B,)`` with the number of tokens
            actually produced per batch item.

        Raises
        ------
        ValueError
            If input ranks are not 3, if the batch/point dimensions of
            positions and features disagree, or if ``point_counts`` has
            the wrong shape.
        """
        if point_positions.ndim != 3 or point_features.ndim != 3:
            raise ValueError(
                "PointCloudTokenizer.forward_batched expects rank-3 "
                "point_positions and point_features."
            )
        if point_positions.shape[:2] != point_features.shape[:2]:
            raise ValueError(
                "point_positions and point_features must agree on batch/point dims, "
                f"got {point_positions.shape[:2]} and {point_features.shape[:2]}"
            )

        batch_size = int(point_positions.shape[0])
        if point_counts is None:
            point_counts = torch.full(
                (batch_size,),
                int(point_positions.shape[1]),
                device=point_positions.device,
                dtype=torch.long,
            )
        elif point_counts.ndim != 1 or int(point_counts.shape[0]) != batch_size:
            raise ValueError("point_counts must have shape [B].")

        token_positions_list = []
        token_features_list = []
        token_counts = []
        for i in range(batch_size):
            n_i = int(point_counts[i].item())
            pos_i, feat_i = self(
                point_positions=point_positions[i, :n_i],
                point_features=point_features[i, :n_i],
            )
            token_positions_list.append(pos_i)
            token_features_list.append(feat_i)
            token_counts.append(int(pos_i.shape[0]))

        max_tokens = max(token_counts) if token_counts else 0
        coord_dim = int(point_positions.shape[-1])
        feat_dim = int(point_features.shape[-1])
        padded_positions = point_positions.new_zeros(
            (batch_size, max_tokens, coord_dim)
        )
        padded_features = point_features.new_zeros(
            (batch_size, max_tokens, feat_dim)
        )
        token_mask = torch.zeros(
            (batch_size, max_tokens),
            device=point_positions.device,
            dtype=torch.bool,
        )

        for i, (pos_i, feat_i, count_i) in enumerate(
            zip(token_positions_list, token_features_list, token_counts, strict=False)
        ):
            if count_i <= 0:
                continue
            padded_positions[i, :count_i] = pos_i
            padded_features[i, :count_i] = feat_i
            token_mask[i, :count_i] = True

        token_counts_t = torch.tensor(
            token_counts, device=point_positions.device, dtype=torch.long
        )
        return padded_positions, padded_features, token_mask, token_counts_t
