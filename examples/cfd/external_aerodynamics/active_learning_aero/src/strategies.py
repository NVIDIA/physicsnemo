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
