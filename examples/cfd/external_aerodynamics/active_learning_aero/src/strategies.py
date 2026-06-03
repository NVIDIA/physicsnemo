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

"""Active learning strategies for GeoTransolver + GP aerodynamics.

Provides query, label, and metrology strategies for the active learning
loop that selects the most informative DrivAerStar geometries.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.distributed import DistributedManager
from physicsnemo.active_learning.protocols import (
    AbstractQueue,
    ActiveLearningPhase,
    LabelStrategy,
    QueryStrategy,
)
from physicsnemo.experimental.guardrails.embedded import OODGuard

from utils import cast_precisions, padded_all_gather
from aero_physics import (
    DRAG_COEFF_SCALE,
    compute_drag_from_subsampled_outputs,
)


class JointUQQueryStrategy(QueryStrategy):
    """Select samples with highest joint UQ = max(|disagreement|, 2*GP_std).

    Runs the GeoTransolver + GP inference pipeline on every unlabeled
    sample and ranks by the combined uncertainty signal.

    Parameters
    ----------
    max_samples : int
        Number of samples to select per round.
    precision : str
        Precision for model forward pass (e.g. "float32").
    """

    __protocol_name__ = "JointUQQueryStrategy"
    __protocol_type__ = ActiveLearningPhase.QUERY

    def __init__(self, max_samples: int = 50, precision: str = "float32") -> None:
        self.max_samples = max_samples
        self.precision = precision
        self.driver = None
        self.selection_history: list[dict[str, Any]] = []

    def attach(self, other: object) -> None:
        """Attach this strategy to its driver (called by the AL framework)."""
        self.driver = other

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    @torch.no_grad()
    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
        """Score unlabeled samples by joint UQ across all ranks, enqueue top-N."""
        pool = self.driver.training_pool
        unlabeled = pool.unlabeled_indices()

        if len(unlabeled) == 0:
            self.logger.warning("No unlabeled samples remaining.")
            return

        model = self.driver.learner
        gp = kwargs.get("gp_head")
        embedding_reduction = kwargs.get("embedding_reduction")
        surface_factors = kwargs.get("surface_factors")
        device = kwargs.get("device", torch.device("cuda"))
        dm = DistributedManager()
        rank, world_size = dm.rank, dm.world_size

        backbone = model.module if hasattr(model, "module") else model
        backbone.eval()
        embedding_reduction.eval()
        gp.eval()

        my_indices = unlabeled[rank::world_size]
        n_total = len(unlabeled)
        local_rows = []

        for ui, flat_idx in enumerate(my_indices):
            if ui % 50 == 0 and rank == 0:
                self.logger.info(f"  UQ scoring: ~{ui * world_size}/{n_total}")
            flat_idx = flat_idx.item()
            batch = pool.get_by_flat_idx(flat_idx)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            features = cast_precisions(batch["fx"], self.precision)
            embeddings = cast_precisions(batch["embeddings"], self.precision)
            geometry = (
                cast_precisions(batch["geometry"], self.precision)
                if "geometry" in batch
                else None
            )
            local_positions = embeddings[:, :, :3]

            outputs, embedding_states = backbone(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
                return_embedding_states=True,
            )
            reduced = embedding_reduction(embedding_states.flatten(1, 2))

            mean_scaled, var_scaled, _, _ = gp.predict(reduced)
            gp_std = torch.sqrt(var_scaled).item() * DRAG_COEFF_SCALE
            gp_mean = mean_scaled.item() * DRAG_COEFF_SCALE

            if "surface_areas_sub" in batch and "surface_normals_sub" in batch:
                trans_cd = (
                    compute_drag_from_subsampled_outputs(
                        outputs, batch, surface_factors, device
                    ).item()
                    * DRAG_COEFF_SCALE
                )
                disagreement = abs(gp_mean - trans_cd)
            else:
                disagreement = 0.0

            joint_uq = max(disagreement, 2.0 * gp_std)
            local_rows.append([float(flat_idx), joint_uq, disagreement, gp_std])

        # 4 columns: (flat_idx, joint_uq, disagreement, gp_std). Empty list
        # must be a (0, 4) tensor, not (0,), so the gather sees consistent shape.
        if local_rows:
            local_t = torch.tensor(local_rows, dtype=torch.float64, device=device)
        else:
            local_t = torch.zeros((0, 4), dtype=torch.float64, device=device)
        all_data = padded_all_gather(local_t, device).cpu().numpy()

        scores = [
            (int(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for row in all_data
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[: self.max_samples]

        round_record = {
            "selected": [],
            "step": getattr(self.driver, "active_learning_step_idx", -1),
        }
        for flat_idx, uq, dis, std in selected:
            query_queue.put(flat_idx)
            round_record["selected"].append(
                {
                    "flat_idx": flat_idx,
                    "class": pool.class_of(flat_idx),
                    "joint_uq": float(uq),
                    "disagreement": float(dis),
                    "gp_std": float(std),
                }
            )
        self.selection_history.append(round_record)

        if rank == 0:
            class_counts = defaultdict(int)
            for entry in round_record["selected"]:
                class_counts[entry["class"]] += 1
            self.logger.info(f"Selected {len(selected)} samples: {dict(class_counts)}")


class RandomQueryStrategy(QueryStrategy):
    """Uniform random selection from the unlabeled pool (baseline).

    Parameters
    ----------
    max_samples : int
        Number of samples to select per round.
    seed : int | None
        Random seed for reproducibility.
    """

    __protocol_name__ = "RandomQueryStrategy"
    __protocol_type__ = ActiveLearningPhase.QUERY

    def __init__(self, max_samples: int = 50, seed: int | None = None) -> None:
        self.max_samples = max_samples
        self.seed = seed
        self.driver = None
        self._rng = np.random.default_rng(seed)
        self.selection_history: list[dict[str, Any]] = []

    def attach(self, other: object) -> None:
        """Attach this strategy to its driver (called by the AL framework)."""
        self.driver = other

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
        """Pick ``max_samples`` indices uniformly at random from the unlabeled pool."""
        pool = self.driver.training_pool
        unlabeled = pool.unlabeled_indices().numpy()

        n = min(self.max_samples, len(unlabeled))
        if n == 0:
            return

        chosen = self._rng.choice(unlabeled, size=n, replace=False)

        round_record = {
            "selected": [],
            "step": getattr(self.driver, "active_learning_step_idx", -1),
        }
        for flat_idx in chosen:
            flat_idx = int(flat_idx)
            query_queue.put(flat_idx)
            round_record["selected"].append(
                {
                    "flat_idx": flat_idx,
                    "class": pool.class_of(flat_idx),
                }
            )
        self.selection_history.append(round_record)

        class_counts = defaultdict(int)
        for entry in round_record["selected"]:
            class_counts[entry["class"]] += 1
        self.logger.info(f"Randomly selected {n} samples: {dict(class_counts)}")


class ClassBalancedRandomQueryStrategy(QueryStrategy):
    """Stratified random selection: equal-as-possible per class from the unlabeled pool.

    For pools with K classes and ``max_samples=N``, this picks roughly
    ``N // K`` samples per class. Any remainder is distributed deterministically
    across classes in sorted-name order so that all DDP ranks compute the
    same target counts. If a class lacks enough unlabeled samples to meet its
    target, the deficit is redistributed to other classes that still have
    headroom.

    Useful as a fairer baseline than uniform random when the underlying pool
    is class-imbalanced or when one wants to test whether UQ-driven acquisition
    contributes anything beyond enforced class balancing.

    Parameters
    ----------
    max_samples : int
        Number of samples to select per round.
    seed : int | None
        Random seed for reproducibility (shared across DDP ranks).
    """

    __protocol_name__ = "ClassBalancedRandomQueryStrategy"
    __protocol_type__ = ActiveLearningPhase.QUERY

    def __init__(self, max_samples: int = 50, seed: int | None = None) -> None:
        self.max_samples = max_samples
        self.seed = seed
        self.driver = None
        self._rng = np.random.default_rng(seed)
        self.selection_history: list[dict[str, Any]] = []

    def attach(self, other: object) -> None:
        """Attach this strategy to its driver (called by the AL framework)."""
        self.driver = other

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
        """Sample ``max_samples`` indices balanced across class labels."""
        pool = self.driver.training_pool
        unlabeled = pool.unlabeled_indices().numpy()

        if len(unlabeled) == 0:
            return

        buckets: dict[str, list[int]] = defaultdict(list)
        for idx in unlabeled:
            buckets[pool.class_of(int(idx))].append(int(idx))

        classes = sorted(buckets.keys())
        n_classes = len(classes)

        base = self.max_samples // n_classes
        remainder = self.max_samples - base * n_classes
        targets = {c: base + (1 if i < remainder else 0) for i, c in enumerate(classes)}

        picks_by_class: dict[str, list[int]] = {}
        deficit = 0
        for c in classes:
            n_avail = len(buckets[c])
            n_want = targets[c]
            if n_avail <= n_want:
                picks_by_class[c] = list(buckets[c])
                deficit += n_want - n_avail
            else:
                idx_arr = self._rng.choice(buckets[c], size=n_want, replace=False)
                picks_by_class[c] = [int(x) for x in idx_arr]

        # Redistribute deficit deterministically across classes that still
        # have unselected unlabeled samples.
        while deficit > 0:
            progressed = False
            for c in classes:
                if deficit == 0:
                    break
                already = set(picks_by_class[c])
                remaining = [i for i in buckets[c] if i not in already]
                if remaining:
                    extra = self._rng.choice(remaining, size=1, replace=False)
                    picks_by_class[c].append(int(extra[0]))
                    deficit -= 1
                    progressed = True
            if not progressed:
                break

        chosen: list[int] = []
        for c in classes:
            chosen.extend(picks_by_class[c])

        round_record = {
            "selected": [],
            "step": getattr(self.driver, "active_learning_step_idx", -1),
            "targets": targets,
        }
        for flat_idx in chosen:
            query_queue.put(int(flat_idx))
            round_record["selected"].append(
                {
                    "flat_idx": int(flat_idx),
                    "class": pool.class_of(int(flat_idx)),
                }
            )
        self.selection_history.append(round_record)

        class_counts = defaultdict(int)
        for entry in round_record["selected"]:
            class_counts[entry["class"]] += 1
        self.logger.info(
            f"Class-balanced random selected {len(chosen)} samples: "
            f"{dict(class_counts)} (target: {targets})"
        )


class LatentNoveltyQueryStrategy(QueryStrategy):
    """Select unlabeled samples whose learned latent is farthest from the labeled set.

    At each round we (a) calibrate a fresh :class:`OODGuard` on the
    *currently labeled* training pool by collecting per-sample reduced
    embeddings (the same vector the GP head consumes), then (b) score
    every unlabeled sample by its average kNN cosine distance to that
    calibration buffer and rank by descending novelty.  The intuition
    is that the most informative geometries to label next are those the
    model has not seen anything close to yet.

    This strategy is orthogonal to UQ-driven acquisition: it provides a
    geometry-novelty signal that depends only on the encoder's learned
    representation, not on the GP's posterior or the transformer's
    direct drag prediction.

    Implementation notes
    --------------------
    * Calibration is performed redundantly across DDP ranks (each rank
      receives every labeled latent via :func:`padded_all_gather`) so
      that all ranks hold an identical ``OODGuard`` buffer; this keeps
      novelty scores deterministic regardless of rank topology.
    * Scoring is parallelised: each rank scores its slice of the
      unlabeled pool, then all per-sample ``(flat_idx, novelty)`` rows
      are gathered and ranked once.

    Cold start
    ----------
    The very first AL round starts with an empty labeled pool, so there
    is nothing to calibrate the OOD guard against.  In that case the
    strategy delegates to an internal :class:`ClassBalancedRandomQueryStrategy`
    so the round still produces a sensible, class-balanced seed batch.
    The selection_history entry is tagged with ``cold_start_fallback=True``
    so downstream analysis can distinguish bootstrap rounds from
    novelty-driven ones.

    Parameters
    ----------
    max_samples : int
        Number of samples to select per round.
    precision : str
        Precision for model forward pass (e.g. "float32").
    knn_k : int
        Number of nearest neighbours used by the OOD guard when
        computing the average kNN distance.  Clamped at scoring time
        to the calibration buffer size, so values larger than the
        labeled pool are safe.
    cold_start_seed : int | None
        Random seed for the class-balanced random fallback used when
        the labeled pool is empty.  Shared across DDP ranks.
    """

    __protocol_name__ = "LatentNoveltyQueryStrategy"
    __protocol_type__ = ActiveLearningPhase.QUERY

    def __init__(
        self,
        max_samples: int = 50,
        precision: str = "float32",
        knn_k: int = 10,
        cold_start_seed: int | None = 0,
    ) -> None:
        self.max_samples = max_samples
        self.precision = precision
        self.knn_k = knn_k
        self.driver = None
        self.selection_history: list[dict[str, Any]] = []
        # Round-1 bootstrap: no labeled samples yet, so novelty is
        # undefined.  Fall back to a class-balanced random pick.
        self._cold_start = ClassBalancedRandomQueryStrategy(
            max_samples=max_samples, seed=cold_start_seed
        )

    def attach(self, other: object) -> None:
        """Attach this strategy (and its cold-start fallback) to the driver."""
        self.driver = other
        self._cold_start.attach(other)

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    @torch.no_grad()
    def _embed_indices(
        self,
        indices: list[int] | np.ndarray | torch.Tensor,
        *,
        backbone: torch.nn.Module,
        embedding_reduction: torch.nn.Module,
        device: torch.device,
        rank: int,
        world_size: int,
        log_prefix: str,
    ) -> torch.Tensor:
        """Forward a list of pool indices and return reduced per-sample latents.

        Sharding:

        * When ``len(indices) >= world_size`` each rank embeds its
          ``indices[rank::world_size]`` slice, then results are
          all-gathered into a single ``(N, D + 1)`` tensor on every
          rank.  Disjoint shards are gathered as-is.
        * When ``len(indices) < world_size`` (early AL rounds where
          the labeled set has fewer samples than the world size),
          every rank redundantly embeds the full list to avoid
          empty-shard ranks.  The gathered tensor is then deduplicated
          by flat index so the calibration buffer reflects each
          labeled sample exactly once.

        The leading column of the returned tensor is the flat index
        (kept as a float to share dtype with the latent values) and the
        remaining columns are the reduced latent vector.
        """
        pool = self.driver.training_pool
        index_list = [
            int(i)
            for i in (
                indices.tolist() if isinstance(indices, torch.Tensor) else list(indices)
            )
        ]
        n_total = len(index_list)
        # Round-robin shard when work is plentiful; replicate across
        # ranks when work is scarce to keep every rank busy and avoid
        # uneven all-gather payloads of unknown column width.
        if n_total >= world_size:
            my_indices = index_list[rank::world_size]
        else:
            my_indices = list(index_list)
        local_rows: list[torch.Tensor] = []

        for ui, flat_idx in enumerate(my_indices):
            if ui % 50 == 0 and rank == 0:
                self.logger.info(f"  {log_prefix}: ~{ui * world_size}/{n_total}")
            flat_idx = int(flat_idx)
            batch = pool.get_by_flat_idx(flat_idx)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            features = cast_precisions(batch["fx"], self.precision)
            embeddings = cast_precisions(batch["embeddings"], self.precision)
            geometry = (
                cast_precisions(batch["geometry"], self.precision)
                if "geometry" in batch
                else None
            )
            local_positions = embeddings[:, :, :3]

            _, embedding_states = backbone(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
                return_embedding_states=True,
            )
            reduced = embedding_reduction(embedding_states.flatten(1, 2))
            # Squeeze the leading batch dim — pool serves one sample at a
            # time — and prepend the flat index so the gathered rows are
            # self-describing and have a non-NaN sentinel column.
            row = torch.cat(
                [
                    torch.tensor([float(flat_idx)], dtype=torch.float32, device=device),
                    reduced.detach().to(torch.float32).flatten(),
                ]
            )
            local_rows.append(row.unsqueeze(0))

        local_t = torch.cat(local_rows, dim=0)
        gathered = padded_all_gather(local_t, device)
        # Dedupe by flat_idx (column 0), preserving first-occurrence order.
        # This is a no-op for the round-robin shard mode (shards are
        # disjoint) and removes the replicated rows produced by the
        # n_total < world_size branch.  Vectorised via torch.unique +
        # scatter_reduce(amin) so the per-row .item() syncs of a Python
        # loop are replaced by a single device→host sync inside unique.
        n_rows = gathered.shape[0]
        flat_ids = gathered[:, 0].long()
        unique_ids, inverse = torch.unique(flat_ids, return_inverse=True)
        row_index = torch.arange(n_rows, device=device)
        first_idx = torch.full(
            (unique_ids.shape[0],), n_rows, dtype=torch.long, device=device
        )
        first_idx.scatter_reduce_(
            0, inverse, row_index, reduce="amin", include_self=True
        )
        keep_mask = torch.zeros(n_rows, dtype=torch.bool, device=device)
        keep_mask[first_idx] = True
        return gathered[keep_mask]

    @torch.no_grad()
    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
        """Calibrate an OODGuard on the labeled set, score the unlabeled pool, take top-N."""
        pool = self.driver.training_pool
        unlabeled = pool.unlabeled_indices()

        if len(unlabeled) == 0:
            self.logger.warning("No unlabeled samples remaining.")
            return

        labeled = pool.train_indices
        if len(labeled) == 0:
            rank = kwargs.get("rank", 0)
            if rank == 0:
                self.logger.info(
                    "Cold start: labeled pool is empty; falling back to "
                    "class-balanced random for this round."
                )
            self._cold_start.sample(query_queue, *args, **kwargs)
            # Mirror the inner record into our own history with a marker
            # so plot_summary / analysis tools can distinguish bootstrap
            # rounds from novelty-driven ones.
            if self._cold_start.selection_history:
                inner = self._cold_start.selection_history[-1]
                self.selection_history.append({**inner, "cold_start_fallback": True})
            return

        model = self.driver.learner
        embedding_reduction = kwargs.get("embedding_reduction")
        device = kwargs.get("device", torch.device("cuda"))
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

        backbone = model.module if hasattr(model, "module") else model
        backbone.eval()
        embedding_reduction.eval()

        # ---- Phase 1: calibrate OODGuard on labeled latents ----
        if rank == 0:
            self.logger.info(
                f"Calibrating OODGuard on {len(labeled)} labeled samples..."
            )
        labeled_table = self._embed_indices(
            labeled,
            backbone=backbone,
            embedding_reduction=embedding_reduction,
            device=device,
            rank=rank,
            world_size=world_size,
            log_prefix="calibration embed",
        )
        # First column is flat_idx; remainder is the reduced latent.
        labeled_latents = labeled_table[:, 1:].contiguous().to(torch.float32)
        latent_dim = labeled_latents.shape[1]

        guard = OODGuard(
            buffer_size=labeled_latents.shape[0],
            geometry_embed_dim=latent_dim,
            knn_k=self.knn_k,
        ).to(device)
        guard.collect(geometry_latent=labeled_latents)

        # ---- Phase 2: score unlabeled pool ----
        if rank == 0:
            self.logger.info(
                f"Scoring {len(unlabeled)} unlabeled samples for latent novelty..."
            )
        my_indices = unlabeled[rank::world_size].tolist()
        n_total = len(unlabeled)
        local_rows: list[list[float]] = []

        for ui, flat_idx in enumerate(my_indices):
            if ui % 50 == 0 and rank == 0:
                self.logger.info(f"  novelty scoring: ~{ui * world_size}/{n_total}")
            flat_idx = int(flat_idx)
            batch = pool.get_by_flat_idx(flat_idx)
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            features = cast_precisions(batch["fx"], self.precision)
            embeddings = cast_precisions(batch["embeddings"], self.precision)
            geometry = (
                cast_precisions(batch["geometry"], self.precision)
                if "geometry" in batch
                else None
            )
            local_positions = embeddings[:, :, :3]
            _, embedding_states = backbone(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
                return_embedding_states=True,
            )
            reduced = embedding_reduction(embedding_states.flatten(1, 2))
            novelty = guard.score_geometry(reduced.detach().to(torch.float32))
            local_rows.append([float(flat_idx), float(novelty.item())])

        # Rows are (flat_idx, novelty_score); ranks with no work after the
        # round-robin slice must contribute an explicit (0, 2) tensor so
        # padded_all_gather sees a consistent column count across ranks
        # (an empty list would otherwise produce shape (0,) and a stray
        # unsqueeze would yield (1, 0), corrupting the gather).
        if len(local_rows) == 0:
            local_t = torch.zeros((0, 2), dtype=torch.float64, device=device)
        else:
            local_t = torch.tensor(local_rows, dtype=torch.float64, device=device)
        all_data = padded_all_gather(local_t, device).cpu().numpy()

        # Unpack each gathered row into (flat_idx, novelty_score).
        scores = [(int(row[0]), float(row[1])) for row in all_data]
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[: self.max_samples]

        round_record = {
            "selected": [],
            "step": getattr(self.driver, "active_learning_step_idx", -1),
            "labeled_buffer_size": int(labeled_latents.shape[0]),
            "knn_k_effective": int(min(self.knn_k, labeled_latents.shape[0])),
        }
        for flat_idx, novelty in selected:
            query_queue.put(flat_idx)
            round_record["selected"].append(
                {
                    "flat_idx": flat_idx,
                    "class": pool.class_of(flat_idx),
                    "latent_novelty": float(novelty),
                }
            )
        self.selection_history.append(round_record)

        if rank == 0:
            class_counts = defaultdict(int)
            for entry in round_record["selected"]:
                class_counts[entry["class"]] += 1
            self.logger.info(
                f"Selected {len(selected)} samples by latent novelty: "
                f"{dict(class_counts)} "
                f"(buffer={round_record['labeled_buffer_size']}, "
                f"k={round_record['knn_k_effective']})"
            )


class DummyLabelStrategy(LabelStrategy):
    """Pass-through: labels already exist in the dataset.

    Simply moves indices from the query queue to the label queue.
    """

    __protocol_name__ = "DummyLabelStrategy"
    __protocol_type__ = ActiveLearningPhase.LABELING
    __is_external_process__ = False
    __provides_fields__ = None

    def __init__(self) -> None:
        self.driver = None

    def attach(self, other: object) -> None:
        """Attach this strategy to its driver (called by the AL framework)."""
        self.driver = other

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    def label(
        self,
        queue_to_label: AbstractQueue,
        serialize_queue: AbstractQueue,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """Pass-through label: forward every queried item to the serialize queue."""
        while not queue_to_label.empty():
            item = queue_to_label.get()
            serialize_queue.put(item)
