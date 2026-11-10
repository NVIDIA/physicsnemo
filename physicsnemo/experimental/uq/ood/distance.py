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

"""Distance-based novelty scoring for caller-provided feature tensors."""

from __future__ import annotations

import torch
from jaxtyping import Float


def _knn_score(
    reference, query, k, query_chunk_size, reference_chunk_size, *, exclude_self=False
):
    """Exact neighbors with bounded distance blocks and optional diagonal exclusion."""
    k = min(k, reference.shape[0] - int(exclude_self))
    scores = query.new_empty(query.shape[0])
    for q_start in range(0, query.shape[0], query_chunk_size):
        q = query[q_start : q_start + query_chunk_size]
        nearest = q.new_empty((q.shape[0], 0))
        for r_start in range(0, reference.shape[0], reference_chunk_size):
            r = reference[r_start : r_start + reference_chunk_size]
            distances = torch.cdist(q, r, compute_mode="donot_use_mm_for_euclid_dist")
            if exclude_self:
                start, stop = (
                    max(q_start, r_start),
                    min(q_start + len(q), r_start + len(r)),
                )
                indices = (
                    torch.arange(start, stop, device=q.device) if start < stop else None
                )
                if indices is not None:
                    distances[indices - q_start, indices - r_start] = float("inf")
            candidates = torch.cat((nearest, distances), dim=1)
            nearest = candidates.topk(
                min(k, candidates.shape[1]), dim=1, largest=False
            ).values
        scores[q_start : q_start + len(q)] = nearest.mean(dim=1)
    return scores


def _validate_features(features, name, *, allow_empty=False):
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(f"{name} must be a tensor with shape [samples, features].")
    if features.shape[1] == 0 or (not allow_empty and features.shape[0] == 0):
        raise ValueError(f"{name} must have nonzero sample and feature dimensions.")
    if features.dtype not in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
        torch.float64,
    ):
        raise ValueError(f"{name} must use a real floating-point dtype.")
    if not torch.isfinite(features).all():
        raise ValueError(f"{name} must contain only finite values.")


class KNNScorer:
    r"""Mean distance to the nearest reference features.

    Parameters
    ----------
    k : int, default=10
        Neighbors per query, capped at the number of reference samples.
    normalize : bool, default=True
        Divide each vector by its L2 norm plus ``1e-8`` before scoring, matching
        the former embedding guard. Scores are Euclidean distances between
        these vectors, not cosine distances. Zero vectors remain zero.
    query_chunk_size : int, default=256
        Maximum queries in a distance block.
    reference_chunk_size : int, default=4096
        Maximum reference rows in a distance block. Chunk sizes control runtime
        memory use and are not included in fitted state.

    Notes
    -----
    Larger scores indicate greater novelty. The caller owns feature extraction,
    thresholds, and decisions. Fitting copies detached reference features;
    half precision is promoted to float32 and float64 is preserved. Query
    features use the fitted dtype and must be on the same device. Exact torch
    distances are evaluated in blocks on CPU and CUDA, without a cuML index.

    :meth:`score` includes self-neighbors. Use a separate validation set or
    :meth:`reference_scores` for leave-one-out threshold selection. Neither
    approach by itself guarantees a false-positive rate. Save feature identity
    (checkpoint, pooling, schema, units) beside
    :meth:`state_dict` and check it before loading. Shape alone does not establish
    that two feature representations are compatible.

    Examples
    --------
    >>> scorer = KNNScorer(k=1, normalize=False)
    >>> scorer.fit(torch.tensor([[0., 0.], [2., 0.]]))
    >>> scorer.score(torch.tensor([[1., 0.]]))
    tensor([1.])
    """

    def __init__(
        self,
        k: int = 10,
        normalize: bool = True,
        *,
        query_chunk_size: int = 256,
        reference_chunk_size: int = 4096,
    ):
        if isinstance(k, bool) or not isinstance(k, int) or k < 1:
            raise ValueError("k must be a positive integer.")
        if not isinstance(normalize, bool):
            raise ValueError("normalize must be a bool.")
        self.k = k
        self.normalize = normalize
        for name, value in (
            ("query_chunk_size", query_chunk_size),
            ("reference_chunk_size", reference_chunk_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.query_chunk_size = query_chunk_size
        self.reference_chunk_size = reference_chunk_size
        self._reference = None
        self._normalized_reference = None

    @torch.no_grad()
    def fit(self, features: Float[torch.Tensor, "samples features"]) -> None:
        """Replace the reference population with a detached copy of features."""
        _validate_features(features, "Reference features")
        dtype = torch.float64 if features.dtype == torch.float64 else torch.float32
        self._reference = features.detach().to(dtype=dtype).clone()
        self._normalized_reference = self._reference / (
            self._reference.norm(dim=-1, keepdim=True) + 1e-8
        )

    @torch.no_grad()
    def score(
        self, features: Float[torch.Tensor, "samples features"]
    ) -> Float[torch.Tensor, "samples"]:  # noqa: F821
        """Return one novelty score per query without changing the fitted state."""
        if self._reference is None:
            raise RuntimeError("Call fit before scoring.")
        _validate_features(features, "Query features", allow_empty=True)
        if features.shape[1] != self._reference.shape[1]:
            raise ValueError("Query feature dimension must match the fitted reference.")
        if features.device != self._reference.device:
            raise ValueError("Query and reference features must be on the same device.")
        if features.shape[0] == 0:
            return self._reference.new_empty((0,))
        query = features.detach().to(self._reference.dtype)
        if not torch.isfinite(query).all():
            raise ValueError("Query features overflow the fitted dtype.")
        if self.normalize:
            query = query / (query.norm(dim=-1, keepdim=True) + 1e-8)
        return _knn_score(
            self._normalized_reference if self.normalize else self._reference,
            query,
            self.k,
            self.query_chunk_size,
            self.reference_chunk_size,
        )

    @torch.no_grad()
    def reference_scores(self) -> Float[torch.Tensor, "samples"]:  # noqa: F821
        """Score each reference row against the other reference rows.

        Returns
        -------
        torch.Tensor
            Leave-one-out mean distances, using ``min(k, N - 1)`` neighbors.
            Only the row itself is excluded; duplicate rows remain neighbors.

        Raises
        ------
        RuntimeError
            If the scorer has not been fitted.
        ValueError
            If fewer than two reference rows are available.
        """
        if self._reference is None:
            raise RuntimeError("Call fit before scoring.")
        if len(self._reference) < 2:
            raise ValueError(
                "Leave-one-out scoring requires at least two reference rows."
            )
        reference = self._normalized_reference if self.normalize else self._reference
        return _knn_score(
            reference,
            reference,
            self.k,
            self.query_chunk_size,
            self.reference_chunk_size,
            exclude_self=True,
        )

    def state_dict(self) -> dict:
        """Return an independent snapshot supported by weights-only torch.load.

        Store application feature metadata and decision thresholds alongside
        this dictionary. The snapshot includes the scoring configuration.
        """
        if self._reference is None:
            raise RuntimeError("Call fit before saving the scorer.")
        return {
            "version": 1,
            "k": self.k,
            "normalize": self.normalize,
            "reference": self._reference.detach().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore a fitted snapshot, including k and normalization settings.

        Use ``torch.load(..., weights_only=True, map_location=device)`` to
        select the device. Invalid states leave the existing scorer unchanged.
        """
        if set(state) != {"version", "k", "normalize", "reference"}:
            raise ValueError("Invalid KNNScorer state keys.")
        if type(state["version"]) is not int or state["version"] != 1:
            raise ValueError("Unsupported KNNScorer state version.")
        candidate = KNNScorer(k=state["k"], normalize=state["normalize"])
        candidate.fit(state["reference"])
        self.k, self.normalize = candidate.k, candidate.normalize
        self._reference = candidate._reference
        self._normalized_reference = candidate._normalized_reference
