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

Provides ``FieldMetrologyStrategy``: a metrology strategy that
evaluates surface-field MSE on a fixed validation pool, broken down
by dataset class. Lives in this aero-flavoured module because the
per-sample loop knows the AeroDataPool batch layout (``fx``,
``embeddings``, ``geometry``, ``fields``); to adapt to a different
dataset, replace this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from physicsnemo.distributed import DistributedManager
from physicsnemo.active_learning.protocols import (
    ActiveLearningPhase,
    MetrologyStrategy,
)

from utils import cast_precisions, padded_all_gather


class FieldMetrologyStrategy(MetrologyStrategy):
    """Evaluate per-class surface-field MSE on a fixed validation pool.

    DDP-parallel: each rank processes a strided slice of the validation
    pool, then results are all-gathered for a global summary. The
    output record per AL round is::

        {
            "step": <al_round>,
            "n_train": <num training samples>,
            "field_mse": <global mean field MSE>,
            "per_class_field_mse": {<class_label>: <mean field MSE>},
        }

    Parameters
    ----------
    precision : str
        Model precision used during the evaluation forward pass.
    chunk_size : int
        Reserved for future chunked inference; currently unused.
    """

    __protocol_name__ = "FieldMetrologyStrategy"
    __protocol_type__ = ActiveLearningPhase.METROLOGY

    def __init__(self, precision: str = "float32", chunk_size: int = 51200) -> None:
        self.precision = precision
        self.chunk_size = chunk_size
        self.records: list[dict[str, Any]] = []
        self.driver = None

    def attach(self, other: object) -> None:
        """Attach this strategy to its driver (called by the AL framework)."""
        self.driver = other

    @property
    def is_attached(self) -> bool:
        """Return True once a driver has been attached."""
        return self.driver is not None

    @torch.no_grad()
    def compute(self, *args: Any, **kwargs: Any) -> None:
        """Run DDP-parallel field-MSE evaluation on the validation pool."""
        val_pool = self.driver.validation_pool
        model = self.driver.learner
        device = kwargs.get("device", torch.device("cuda"))
        dm = DistributedManager()
        rank, world_size = dm.rank, dm.world_size
        n_train = len(self.driver.training_pool)

        backbone = model.module if hasattr(model, "module") else model
        backbone.eval()

        n_val = len(val_pool)
        my_indices = list(range(rank, n_val, world_size))
        # Materialize train_indices once as a Python list so the inner loop
        # uses CPU integer lookups rather than per-iteration tensor.item().
        train_idx_list = val_pool.train_indices.tolist()

        # Build class<->index map dynamically from the validation pool so the
        # metrology works for any set of class labels (F/N/E, SE/SF, etc.).
        unique_classes = sorted(set(val_pool.class_labels))
        cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
        idx_to_cls = {i: c for c, i in cls_to_idx.items()}

        local_rows = []
        for count, i in enumerate(my_indices):
            if count % 10 == 0 and rank == 0:
                self.logger.info(f"  Metrology: ~{count * world_size}/{n_val}")
            flat_idx = train_idx_list[i]
            batch = val_pool.get_by_flat_idx(flat_idx)
            cls_label = val_pool.class_of(flat_idx)
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

            outputs, _ = backbone(
                global_embedding=features,
                local_embedding=embeddings,
                geometry=geometry,
                local_positions=local_positions,
                return_embedding_states=True,
            )

            field_mse = F.mse_loss(outputs, batch["fields"]).item()
            cls_idx = cls_to_idx.get(cls_label, -1)
            local_rows.append([field_mse, float(cls_idx)])

        # 2 columns: (field_mse, cls_idx). Empty list must be a (0, 2) tensor,
        # not (0,), so the gather sees consistent shape across ranks.
        if local_rows:
            local_t = torch.tensor(local_rows, dtype=torch.float64, device=device)
        else:
            local_t = torch.zeros((0, 2), dtype=torch.float64, device=device)
        all_data = padded_all_gather(local_t, device).cpu().numpy()

        mse_arr = all_data[:, 0]
        cls_arr = all_data[:, 1].astype(int)

        per_class_field_mse = {}
        for ci, cls_label in idx_to_cls.items():
            mask = cls_arr == ci
            if mask.sum() == 0:
                continue
            per_class_field_mse[cls_label] = float(np.mean(mse_arr[mask]))

        step = getattr(self.driver, "active_learning_step_idx", -1)
        record = {
            "step": step,
            "n_train": n_train,
            "field_mse": float(np.mean(mse_arr)),
            "per_class_field_mse": per_class_field_mse,
        }
        self.records.append(record)
        if rank == 0:
            self.logger.info(
                f"Step {step} | n_train={n_train} | "
                f"field_MSE={np.mean(mse_arr):.6f} | "
                f"per_class_fmse={per_class_field_mse}"
            )

    def serialize_records(
        self, path: Path | None = None, *args: Any, **kwargs: Any
    ) -> None:
        """Persist accumulated validation records to JSON."""
        if path is None:
            path = self.strategy_dir / "validation_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.records, f, indent=2)

    def load_records(self, path: Path | None = None, *args: Any, **kwargs: Any) -> None:
        """Load previously serialized validation records from JSON."""
        if path is None:
            path = self.strategy_dir / "validation_metrics.json"
        if path.exists():
            with open(path) as f:
                self.records = json.load(f)
