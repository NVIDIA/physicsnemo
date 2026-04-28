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

"""OOD (Out-of-Distribution) Guard for runtime anomaly detection.

Provides two complementary checks:

1. **Global parameter bounds** — per-channel bounding box on an arbitrary-rank
   global embedding tensor with channel as its last dimension.
2. **Geometry latent kNN** — k-nearest-neighbour distance in a user-provided
   fixed-dimensional latent space.

During training, the guard collects calibration statistics.  During inference,
it compares incoming data against those statistics and emits warnings when
inputs fall outside the training distribution.

The guard is intentionally model-agnostic: callers are responsible for pooling
any higher-rank latent tensor down to ``(B, D)`` before passing it in.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn

from physicsnemo.nn.functional import knn

logger = logging.getLogger(__name__)


_RED = "\033[91m"
_RESET = "\033[0m"


def _reduce_leading(
    x: torch.Tensor,
    reducer: Callable[..., torch.Tensor],
) -> torch.Tensor:
    """Apply ``reducer`` over all dims except the last (channel) dim."""
    if x.ndim <= 1:
        return x
    reduce_dims = tuple(range(x.ndim - 1))
    return reducer(x, dim=reduce_dims)


@dataclass
class OODGuardConfig:
    """User-facing configuration for :class:`OODGuard`.

    Model-derived fields (``global_dim``, ``geometry_embed_dim``) are supplied
    by the enclosing model and intentionally omitted here.

    Attributes
    ----------
    buffer_size : int
        Capacity of the geometry latent FIFO buffer.  Typically set to the
        training-set size.  No default — callers must pick a value.
    knn_k : int
        Number of nearest neighbours for the geometry kNN distance check.
        Default is ``10``.
    sensitivity : float
        Multiplier on the 99th-percentile kNN distance used as the OOD
        threshold.  Higher values are less sensitive.  Default is ``1.5``.
    """

    buffer_size: int
    knn_k: int = 10
    sensitivity: float = 1.5


class OODGuard(nn.Module):
    """Out-of-distribution guard using global-parameter bounds and geometry kNN.

    Parameters
    ----------
    buffer_size : int
        Capacity of the geometry latent FIFO buffer (typically = training set size).
    global_dim : int | None
        Channel dimension of global embeddings.  ``None`` disables the global check.
    geometry_embed_dim : int | None
        Dimensionality of the pooled geometry latent vector.  ``None`` disables
        the geometry kNN check.
    knn_k : int
        Number of nearest neighbours for the geometry distance check.
    sensitivity : float
        Multiplier on the 99th-percentile kNN distance used as the OOD
        threshold.  Higher values are less sensitive.  Default is ``1.5``.
    """

    def __init__(
        self,
        buffer_size: int,
        global_dim: int | None = None,
        geometry_embed_dim: int | None = None,
        knn_k: int = 10,
        sensitivity: float = 1.5,
    ) -> None:
        super().__init__()
        self.buffer_size = buffer_size
        self.sensitivity = sensitivity

        # Global parameter bounds
        if global_dim is not None:
            self.register_buffer(
                "global_min", torch.full((global_dim,), float("inf"))
            )
            self.register_buffer(
                "global_max", torch.full((global_dim,), float("-inf"))
            )
        else:
            self.register_buffer("global_min", None)
            self.register_buffer("global_max", None)

        # Geometry kNN buffer
        if geometry_embed_dim is not None:
            self.register_buffer(
                "geo_embeddings", torch.zeros(buffer_size, geometry_embed_dim)
            )
            # Write index into the FIFO, kept in [0, buffer_size).
            self.register_buffer("geo_ptr", torch.zeros(1, dtype=torch.long))
            # Latches True once the FIFO has been filled at least once.
            self.register_buffer("geo_full", torch.zeros(1, dtype=torch.bool))
            self.register_buffer("knn_threshold", torch.tensor(float("inf")))
        else:
            self.register_buffer("geo_embeddings", None)
            self.register_buffer("geo_ptr", None)
            self.register_buffer("geo_full", None)
            self.register_buffer("knn_threshold", None)

        self.knn_k = knn_k
        self._threshold_stale = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def collect(
        self,
        global_embedding: torch.Tensor | None = None,
        geometry_latent: torch.Tensor | None = None,
    ) -> None:
        """Accumulate calibration data (call during training).

        Parameters
        ----------
        global_embedding : Tensor | None
            Shape ``(B, ..., C_g)`` — at least one leading (batch) dim;
            last dim is channel.
        geometry_latent : Tensor | None
            Shape ``(B, D)`` — pre-pooled per-sample geometry latent vector.
        """
        self._validate_shapes(global_embedding, geometry_latent)
        self._collect_global(global_embedding)
        self._collect_geometry(geometry_latent)
        # Any new geometry sample invalidates the kNN threshold.
        if geometry_latent is not None and self.geo_embeddings is not None:
            self._threshold_stale = True

    @torch.no_grad()
    def check(
        self,
        global_embedding: torch.Tensor | None = None,
        geometry_latent: torch.Tensor | None = None,
    ) -> None:
        """Run OOD checks and emit warnings (call during inference).

        Parameters
        ----------
        global_embedding : Tensor | None
            Shape ``(B, ..., C_g)`` — at least one leading (batch) dim;
            last dim is channel.
        geometry_latent : Tensor | None
            Shape ``(B, D)`` — pre-pooled per-sample geometry latent vector.
        """
        self._validate_shapes(global_embedding, geometry_latent)
        self._check_global(global_embedding)
        # Lazy threshold computation on first inference call
        if self._threshold_stale:
            self.compute_threshold()
            self._threshold_stale = False
        self._check_geometry(geometry_latent)

    @torch.compiler.disable
    @torch.no_grad()
    def compute_threshold(self) -> None:
        """Compute the kNN threshold from the accumulated geometry buffer."""
        if self.geo_embeddings is None:
            return
        n_valid = self._n_valid()
        if n_valid == 0:
            return
        store = self.geo_embeddings[:n_valid]
        store_norm = store / (store.norm(dim=-1, keepdim=True) + 1e-8)
        k = min(self.knn_k, n_valid - 1)
        if k <= 0:
            return
        # Leave-one-out: ask for k+1 neighbours and drop column 0 (each
        # point's nearest neighbour is itself, distance 0).
        _, dists = knn(store_norm, store_norm, k + 1)
        avg_knn_dists = dists[:, 1:].mean(dim=-1)
        base = torch.quantile(avg_knn_dists, 0.99)
        threshold = base * self.sensitivity
        self.knn_threshold.copy_(threshold)
        logger.info(
            "OOD Guard: computed kNN threshold=%.4f (base_99pct=%.4f, sensitivity=%.2f, k=%d)",
            threshold.item(),
            base.item(),
            self.sensitivity,
            self.knn_k,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _n_valid(self) -> int:
        """Number of populated rows in the geometry FIFO buffer."""
        if self.geo_full.item():
            return self.buffer_size
        return self.geo_ptr.item()

    def _validate_shapes(
        self,
        global_embedding: torch.Tensor | None,
        geometry_latent: torch.Tensor | None,
    ) -> None:
        """Validate caller-supplied tensor shapes against the guard's config.

        Skipped under ``torch.compile`` to avoid graph breaks on shape checks.
        """
        if torch.compiler.is_compiling():
            return
        if global_embedding is not None and self.global_min is not None:
            if global_embedding.ndim < 2:
                raise ValueError(
                    f"global_embedding must have at least 2 dims "
                    f"(batch + channel); got {global_embedding.ndim}D tensor "
                    f"with shape {tuple(global_embedding.shape)}. Did you mean "
                    f"to unsqueeze a batch dim?"
                )
            expected = self.global_min.shape[0]
            got = global_embedding.shape[-1]
            if got != expected:
                raise ValueError(
                    f"global_embedding last-dim mismatch: expected {expected} "
                    f"(from global_dim), got {got} "
                    f"(shape {tuple(global_embedding.shape)})"
                )
        if geometry_latent is not None and self.geo_embeddings is not None:
            if geometry_latent.ndim != 2:
                raise ValueError(
                    f"geometry_latent must be rank-2 (B, D); got "
                    f"{geometry_latent.ndim}D tensor with shape "
                    f"{tuple(geometry_latent.shape)}. Pool any higher-rank "
                    f"latent at the caller before passing it in."
                )
            expected = self.geo_embeddings.shape[1]
            got = geometry_latent.shape[1]
            if got != expected:
                raise ValueError(
                    f"geometry_latent channel dim mismatch: expected "
                    f"{expected} (from geometry_embed_dim), got {got}"
                )

    def _collect_global(self, global_embedding: torch.Tensor | None) -> None:
        if global_embedding is None or self.global_min is None:
            return
        # Upcast to the buffer dtype so AMP (fp16/bf16) inputs don't mismatch
        # the fp32 running min/max.
        vals = global_embedding.detach().to(self.global_min.dtype)
        batch_min = _reduce_leading(vals, torch.amin)
        batch_max = _reduce_leading(vals, torch.amax)
        self.global_min.copy_(torch.minimum(self.global_min, batch_min))
        self.global_max.copy_(torch.maximum(self.global_max, batch_max))

    def _collect_geometry(self, geometry_latent: torch.Tensor | None) -> None:
        if geometry_latent is None or self.geo_embeddings is None:
            return
        # Upcast to the buffer dtype so AMP (fp16/bf16) inputs don't fail the
        # dtype-strict indexed assignment into geo_embeddings.
        pooled = geometry_latent.detach().to(self.geo_embeddings.dtype)  # (B, D)
        B = pooled.shape[0]
        ptr = self.geo_ptr[0]
        indices = (ptr + torch.arange(B, device=pooled.device)) % self.buffer_size
        self.geo_embeddings[indices] = pooled
        wrapped = ((ptr + B) >= self.buffer_size).view(1)
        self.geo_full.logical_or_(wrapped)
        self.geo_ptr.fill_((ptr + B) % self.buffer_size)

    @torch.compiler.disable
    def _check_global(self, global_embedding: torch.Tensor | None) -> None:
        if global_embedding is None or self.global_min is None:
            return
        if torch.isinf(self.global_min).any():
            return
        # Upcast so AMP inputs compare against the fp32 bounds cleanly.
        vals = global_embedding.detach().to(self.global_min.dtype)
        batch_min = _reduce_leading(vals, torch.amin)
        batch_max = _reduce_leading(vals, torch.amax)
        below = batch_min < self.global_min
        above = batch_max > self.global_max
        # Skip host transfer when nothing is violated and DEBUG is off.
        if not (bool((below | above).any()) or logger.isEnabledFor(logging.DEBUG)):
            return
        # Single bulk transfer; then iterate in Python over dims.
        bmin_l = batch_min.tolist()
        bmax_l = batch_max.tolist()
        lo_l = self.global_min.tolist()
        hi_l = self.global_max.tolist()
        below_l = below.tolist()
        above_l = above.tolist()
        for d, (bmin, bmax, lo, hi) in enumerate(
            zip(bmin_l, bmax_l, lo_l, hi_l)
        ):
            logger.debug(
                "OOD Guard [global] dim %d: val=[%.4f, %.4f] bounds=[%.4f, %.4f]",
                d, bmin, bmax, lo, hi,
            )
            if below_l[d]:
                logger.warning(
                    f"{_RED}OOD Guard: global_embedding dim {d} value "
                    f"{bmin:.4f} below training min {lo:.4f}{_RESET}"
                )
            if above_l[d]:
                logger.warning(
                    f"{_RED}OOD Guard: global_embedding dim {d} value "
                    f"{bmax:.4f} above training max {hi:.4f}{_RESET}"
                )

    @torch.compiler.disable
    def _check_geometry(self, geometry_latent: torch.Tensor | None) -> None:
        if geometry_latent is None or self.geo_embeddings is None:
            return
        if torch.isinf(self.knn_threshold):
            return
        # Upcast so ``cdist`` against the fp32 store works under AMP inputs.
        pooled = geometry_latent.detach().to(self.geo_embeddings.dtype)  # (B, D)
        z = pooled / (pooled.norm(dim=-1, keepdim=True) + 1e-8)
        n_valid = self._n_valid()
        if n_valid == 0:
            return
        store = self.geo_embeddings[:n_valid]
        store_norm = store / (store.norm(dim=-1, keepdim=True) + 1e-8)
        # Query is not in the store, so no -1 needed; clamp to buffer size.
        k = min(self.knn_k, n_valid)
        _, dists = knn(store_norm, z, k)  # (B, k)
        avg_knn_dists = dists.mean(dim=-1)  # (B,)
        over = avg_knn_dists > self.knn_threshold
        # Skip host transfer when nothing is violated and DEBUG is off.
        if not (bool(over.any()) or logger.isEnabledFor(logging.DEBUG)):
            return
        # Single bulk transfer; then iterate in Python over batch.
        dist_l = avg_knn_dists.tolist()
        over_l = over.tolist()
        threshold = self.knn_threshold.item()
        for i, dist_val in enumerate(dist_l):
            logger.debug(
                "OOD Guard [geometry] sample %d: kNN_dist=%.4f threshold=%.4f",
                i, dist_val, threshold,
            )
            if over_l[i]:
                logger.warning(
                    f"{_RED}OOD Guard: geometry sample {i} kNN distance "
                    f"{dist_val:.4f} above threshold {threshold:.4f}{_RESET}"
                )
