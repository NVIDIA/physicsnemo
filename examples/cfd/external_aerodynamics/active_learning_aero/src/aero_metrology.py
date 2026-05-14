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

"""External-aerodynamics metrology for the active-learning example.

The metrology strategy here is the CFD-specific evaluator that reports
field MSE and drag accuracy on a fixed validation pool. It is split out
from ``strategies.py`` so that the generic strategies (query / label)
do not depend on aero physics. To adapt this example to a different
problem, replace this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.active_learning.protocols import (
    ActiveLearningPhase,
    MetrologyStrategy,
)

from utils import cast_precisions, padded_all_gather
from aero_physics import (
    DRAG_COEFF_SCALE,
    compute_drag_from_subsampled_outputs,
    compute_drag_target_from_batch,
)


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
        all_data = padded_all_gather(local_t, device).cpu().numpy()

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
