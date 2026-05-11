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

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.active_learning.protocols import (
    AbstractQueue,
    ActiveLearningPhase,
    LabelStrategy,
    MetrologyStrategy,
    QueryStrategy,
)

import torch.distributed as dist

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gp_utils import (
    DRAG_COEFF_SCALE,
    cast_precisions,
    compute_drag_from_subsampled_outputs,
    compute_drag_target_from_batch,
)


def _padded_all_gather(local_tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    """All-gather tensors of potentially different row counts across ranks.

    Pads each rank's tensor to the max size, gathers, then strips padding.
    Assumes 2D tensors (N_local, cols). Padding rows are filled with NaN
    so they can be filtered after gathering.
    """
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return local_tensor

    local_size = torch.tensor([local_tensor.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(dist.get_world_size())]
    dist.all_gather(all_sizes, local_size)
    max_size = max(s.item() for s in all_sizes)

    cols = local_tensor.shape[1]
    padded = torch.full((max_size, cols), float("nan"), dtype=local_tensor.dtype, device=device)
    padded[: local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    all_data = torch.cat(gathered, dim=0)

    valid_mask = ~torch.isnan(all_data[:, 0])
    return all_data[valid_mask]


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
        self.driver = other

    @property
    def is_attached(self) -> bool:
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
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)

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
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

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

        local_t = torch.tensor(local_rows, dtype=torch.float64, device=device)
        if local_t.ndim == 1:
            local_t = local_t.unsqueeze(0)
        all_data = _padded_all_gather(local_t, device).cpu().numpy()

        scores = [
            (int(row[0]), float(row[1]), float(row[2]), float(row[3]))
            for row in all_data
        ]
        scores.sort(key=lambda x: x[1], reverse=True)
        selected = scores[: self.max_samples]

        round_record = {"selected": [], "step": getattr(self.driver, "active_learning_step_idx", -1)}
        for flat_idx, uq, dis, std in selected:
            query_queue.put(flat_idx)
            round_record["selected"].append({
                "flat_idx": flat_idx,
                "class": pool.class_of(flat_idx),
                "joint_uq": float(uq),
                "disagreement": float(dis),
                "gp_std": float(std),
            })
        self.selection_history.append(round_record)

        if rank == 0:
            class_counts = defaultdict(int)
            for entry in round_record["selected"]:
                class_counts[entry["class"]] += 1
            self.logger.info(
                f"Selected {len(selected)} samples: {dict(class_counts)}"
            )


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
        self.driver = other

    @property
    def is_attached(self) -> bool:
        return self.driver is not None

    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
        pool = self.driver.training_pool
        unlabeled = pool.unlabeled_indices().numpy()

        n = min(self.max_samples, len(unlabeled))
        if n == 0:
            return

        chosen = self._rng.choice(unlabeled, size=n, replace=False)

        round_record = {"selected": [], "step": getattr(self.driver, "active_learning_step_idx", -1)}
        for flat_idx in chosen:
            flat_idx = int(flat_idx)
            query_queue.put(flat_idx)
            round_record["selected"].append({
                "flat_idx": flat_idx,
                "class": pool.class_of(flat_idx),
            })
        self.selection_history.append(round_record)

        class_counts = defaultdict(int)
        for entry in round_record["selected"]:
            class_counts[entry["class"]] += 1
        self.logger.info(
            f"Randomly selected {n} samples: {dict(class_counts)}"
        )


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
        self.driver = other

    @property
    def is_attached(self) -> bool:
        return self.driver is not None

    def sample(self, query_queue: AbstractQueue, *args: Any, **kwargs: Any) -> None:
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
        targets = {
            c: base + (1 if i < remainder else 0)
            for i, c in enumerate(classes)
        }

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
            round_record["selected"].append({
                "flat_idx": int(flat_idx),
                "class": pool.class_of(int(flat_idx)),
            })
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
        self.driver = other

    @property
    def is_attached(self) -> bool:
        return self.driver is not None

    def label(
        self,
        queue_to_label: AbstractQueue,
        serialize_queue: AbstractQueue,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        while not queue_to_label.empty():
            item = queue_to_label.get()
            serialize_queue.put(item)


class DragMetrologyStrategy(MetrologyStrategy):
    """Evaluate field MSE and drag R^2 on a fixed validation set.

    Parameters
    ----------
    precision : str
        Model precision.
    chunk_size : int
        Chunk size for full-mesh GeoTransolver inference.
    """

    __protocol_name__ = "DragMetrologyStrategy"
    __protocol_type__ = ActiveLearningPhase.METROLOGY

    def __init__(self, precision: str = "float32", chunk_size: int = 51200) -> None:
        self.precision = precision
        self.chunk_size = chunk_size
        self.records: list[dict[str, Any]] = []
        self.driver = None

    def attach(self, other: object) -> None:
        self.driver = other

    @property
    def is_attached(self) -> bool:
        return self.driver is not None

    @torch.no_grad()
    def compute(self, *args: Any, **kwargs: Any) -> None:
        """Run DDP-parallel evaluation on the validation pool."""
        val_pool = self.driver.validation_pool
        model = self.driver.learner
        gp = kwargs.get("gp_head")
        embedding_reduction = kwargs.get("embedding_reduction")
        surface_factors = kwargs.get("surface_factors")
        device = kwargs.get("device", torch.device("cuda"))
        rank = kwargs.get("rank", 0)
        world_size = kwargs.get("world_size", 1)
        n_train = len(self.driver.training_pool)

        backbone = model.module if hasattr(model, "module") else model
        backbone.eval()
        embedding_reduction.eval()
        gp.eval()

        n_val = len(val_pool)
        my_indices = list(range(rank, n_val, world_size))

        # Build class<->index map dynamically from the validation pool so the
        # metrology works for any set of class labels (F/N/E, SE/SF, etc.).
        unique_classes = sorted(set(val_pool.class_labels))
        cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
        idx_to_cls = {i: c for c, i in cls_to_idx.items()}

        local_rows = []
        for count, i in enumerate(my_indices):
            if count % 10 == 0 and rank == 0:
                self.logger.info(f"  Metrology: ~{count * world_size}/{n_val}")
            flat_idx = val_pool.train_indices[i].item()
            batch = val_pool.get_by_flat_idx(flat_idx)
            cls_label = val_pool.class_of(flat_idx)
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

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
            gp_cd = mean_scaled.item() * DRAG_COEFF_SCALE

            trans_cd = 0.0
            if "surface_areas_sub" in batch and "surface_normals_sub" in batch:
                trans_cd = (
                    compute_drag_from_subsampled_outputs(
                        outputs, batch, surface_factors, device
                    ).item()
                    * DRAG_COEFF_SCALE
                )

            target_scaled = compute_drag_target_from_batch(
                batch, surface_factors, device
            )
            true_cd = target_scaled.item() * DRAG_COEFF_SCALE

            field_mse = F.mse_loss(outputs, batch["fields"]).item()

            cls_idx = cls_to_idx.get(cls_label, -1)
            local_rows.append([true_cd, gp_cd, trans_cd, field_mse, float(cls_idx)])

        local_t = torch.tensor(local_rows, dtype=torch.float64, device=device)
        if local_t.ndim == 1:
            local_t = local_t.unsqueeze(0)
        all_data = _padded_all_gather(local_t, device).cpu().numpy()

        true_arr = all_data[:, 0]
        gp_arr = all_data[:, 1]
        trans_arr = all_data[:, 2]
        mse_arr = all_data[:, 3]
        cls_arr = all_data[:, 4].astype(int)

        def _r2(y_true, y_pred):
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - y_true.mean()) ** 2)
            return 1.0 - ss_res / (ss_tot + 1e-12)

        drag_r2_gp = _r2(true_arr, gp_arr)
        drag_r2_trans = _r2(true_arr, trans_arr) if trans_arr.any() else None

        per_class_r2_gp = {}
        per_class_r2_trans = {}
        per_class_field_mse = {}
        for ci, cls_label in idx_to_cls.items():
            mask = cls_arr == ci
            if mask.sum() == 0:
                continue
            t = true_arr[mask]
            per_class_r2_gp[cls_label] = float(_r2(t, gp_arr[mask]))
            if trans_arr.any():
                per_class_r2_trans[cls_label] = float(_r2(t, trans_arr[mask]))
            per_class_field_mse[cls_label] = float(np.mean(mse_arr[mask]))

        step = getattr(self.driver, "active_learning_step_idx", -1)
        record = {
            "step": step,
            "n_train": n_train,
            "drag_r2": float(drag_r2_gp),
            "drag_r2_transolver": float(drag_r2_trans) if drag_r2_trans is not None else None,
            "field_mse": float(np.mean(mse_arr)),
            "per_class_r2": per_class_r2_gp,
            "per_class_r2_transolver": per_class_r2_trans if per_class_r2_trans else None,
            "per_class_field_mse": per_class_field_mse,
        }
        self.records.append(record)
        if rank == 0:
            trans_str = f" | R²_trans={drag_r2_trans:.4f}" if drag_r2_trans is not None else ""
            self.logger.info(
                f"Step {step} | n_train={n_train} | R²_gp={drag_r2_gp:.4f}{trans_str} | "
                f"field_MSE={np.mean(mse_arr):.6f} | "
                f"per_class_gp={per_class_r2_gp} | "
                f"per_class_trans={per_class_r2_trans} | "
                f"per_class_fmse={per_class_field_mse}"
            )

    def serialize_records(
        self, path: Path | None = None, *args: Any, **kwargs: Any
    ) -> None:
        if path is None:
            path = self.strategy_dir / "validation_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def load_records(
        self, path: Path | None = None, *args: Any, **kwargs: Any
    ) -> None:
        if path is None:
            path = self.strategy_dir / "validation_metrics.json"
        if path.exists():
            with open(path) as f:
                self.records = json.load(f)
