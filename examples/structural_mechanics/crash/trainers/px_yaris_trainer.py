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
"""Crash Trainer with exact full-mesh validation for the Yaris recipe.

Training is the unchanged crash loop. Validation differs from the generic
path in two ways that matter for Yaris: every CSV validation case is
evaluated exactly once regardless of the rank count (the generic loader
floors the case count to a multiple of the ranks), and cases have different
horizons (94-152 frames at stride 5), so errors are pooled per rollout step
with explicit counts instead of assuming one horizon. The suite itself lives
in :mod:`metrics.validation_suite` and is shared with ``evaluate_px_yaris.py`` and
``evaluate_biw.py``; this module adds what is Yaris-specific: the split
dataset, the node metadata next to the node maps and the record's
identification keys.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from metrics.validation_suite import (
    ExactShardSampler,
    SuiteResult,
    atomic_json,
    log_summary,
    metric_fields,
    run_suite,
    save_node_rmse,
    training_stats,
)
from train import Trainer

# Reference-x bounds (mm) splitting the full-vehicle mesh into regions (rear,
# middle, front) for the translational-error breakdown.
DEFAULT_REGION_BOUNDS = (-2000.0, -1000.0)
SPLITS = ("validation", "test")


def region_bounds_of(cfg) -> list[float]:
    """``training.validation_region_bounds``; a declared ``null`` means the Yaris default."""
    bounds = cfg.training.get("validation_region_bounds")
    return [float(v) for v in (DEFAULT_REGION_BOUNDS if bounds is None else bounds)]


def build_split_dataset(cfg, reader, split, logger, num_samples=None):
    """The complete trajectories of one manifest split, statistics loaded.

    Built in ``load`` mode: the frozen statistics, the kNN statistics and the
    neighbourhood cache of the work directory are read and never recomputed.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}; got {split!r}")
    names = reader.splits[split]
    if num_samples is None:
        num_samples = len(names)
    if num_samples != len(names):
        raise ValueError(
            f"the {split} split has {len(names)} CSV cases; {num_samples} requested"
        )
    return instantiate(
        cfg.datapipe,
        name=f"crash_{split}",
        reader=reader,
        split=split,
        logger=logger,
        stats_mode="load",
        sample_type="all_time_steps",
        data_dir=cfg.training.raw_data_dir_validation,
        num_samples=num_samples,
    )


def write_node_meta(dataset, path) -> None:
    """Node metadata of the held-out mesh next to the node maps, written once.

    Part ids, the reference positions, the kept-node ids and the first design's
    thickness (it varies per design).
    """
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    reader = dataset.reader
    store = reader.open(dataset.paths[0])
    try:
        part_id = np.asarray(store["nodes/part_id"][:]).reshape(-1)[reader.keep_ids]
    except Exception:  # noqa: BLE001 - metadata is optional
        part_id = np.full(len(reader.keep_ids), -1, dtype=np.int64)
    np.savez(
        path,
        part_id=part_id,
        reference=np.asarray(reader.reference, dtype=np.float32),
        keep_ids=np.asarray(reader.keep_ids, dtype=np.int64),
        thickness=np.asarray(reader.thickness(store), dtype=np.float32),
    )


def yaris_record(result: SuiteResult, dataset, data_stats, epoch: int) -> dict:
    """The Yaris validation record (``epoch_<EEEE>.json``)."""
    reader = dataset.reader
    return dict(
        penetration=result.penetration,
        regions=result.regions,
        node_error_percentiles=result.node_percentiles,
        epoch=int(epoch),
        cases=result.cases,
        kept_nodes=result.kept_nodes,
        excluded_nodes=int(reader.total_nodes - len(reader.keep_ids)),
        metric=(
            "normalized position MSE pooled over every held-out "
            "(case, step, node, channel) entry; outlier nodes excluded"
        ),
        **metric_fields(result, data_stats["node"]["pos_std"], reader.dt),
    )


def evaluate_split(
    model,
    dataset,
    data_stats,
    cfg,
    device,
    *,
    world_size: int = 1,
    rank: int = 0,
    amp: bool = False,
    amp_dtype=None,
    logger=None,
) -> SuiteResult:
    """Run the suite over every case of a Yaris split dataset."""
    return run_suite(
        model,
        dataset,
        [path.stem for path in dataset.paths],
        data_stats,
        int(cfg.training.num_time_steps) - 1,
        device,
        num_nodes=int(dataset.num_nodes),
        world_size=world_size,
        rank=rank,
        amp=amp,
        amp_dtype=amp_dtype,
        penetration_cfg=cfg.training.get("validation_penetration"),
        region_bounds=region_bounds_of(cfg),
        reference_x=torch.as_tensor(dataset.reader.reference[:, 0]),
        node_errors=bool(cfg.training.get("validation_node_errors", False)),
        logger=logger,
    )


def write_outputs(result: SuiteResult, dataset, data_stats, epoch, output_dir):
    """Write the record, the node-error map and the node metadata (rank zero)."""
    output_dir = Path(output_dir)
    if result.node_rmse is not None:
        save_node_rmse(
            output_dir / f"node_rmse_epoch_{epoch:04d}.npy", result.node_rmse
        )
        write_node_meta(dataset, output_dir / "node_meta.npz")
    atomic_json(
        output_dir / f"epoch_{epoch:04d}.json",
        yaris_record(result, dataset, data_stats, epoch),
    )


class YarisFullTrainer(Trainer):
    """Unchanged training loop; exact, horizon-aware held-out validation."""

    def __init__(self, cfg, logger0):
        work_dir = cfg.get("yaris", {}).get("work_dir") if "yaris" in cfg else None
        if work_dir is not None and not Path(str(work_dir)).is_absolute():
            # Hydra changes into the run directory (${yaris.work_dir}); a relative
            # value is then resolved again for stats_dir, ckpt_path and the
            # TensorBoard directory and the outputs nest under themselves.
            raise ValueError(
                f"yaris.work_dir must be an absolute path, got {work_dir!r}"
            )
        # The frame interval is a dataset property (yaris.dt, read by the
        # datapipe); it must agree with the stores' sampling.
        if not math.isclose(
            float(cfg.datapipe.dt),
            float(cfg.reader.raw_dt) * int(cfg.reader.stride),
            abs_tol=1e-12,
        ):
            raise ValueError(
                "datapipe.dt (yaris.dt) differs from reader.raw_dt times yaris.stride"
            )
        requested = int(cfg.training.num_validation_samples)
        # The generic constructor would floor the validation case count to a
        # multiple of the rank count; build validation here instead.
        base_cfg = deepcopy(cfg)
        base_cfg.training.num_validation_samples = 0
        super().__init__(base_cfg, logger0)
        self.cfg = cfg
        self.logger0 = logger0
        self.val_dataloader = None
        self.validation_dataset = None
        output_dir = cfg.training.get("validation_output_dir")
        self.validation_dir = (
            Path(output_dir)
            if output_dir
            else Path(cfg.training.ckpt_path).parent / "validation"
        )
        # The penetration summary of the last validation pass (``None`` without
        # the probe).
        self.last_penetration_summary = None
        if requested <= 0:
            return
        reader = instantiate(cfg.reader)
        names = reader.splits["validation"]
        if requested != len(names):
            raise ValueError(
                f"training.num_validation_samples must equal the {len(names)} CSV "
                f"validation cases, got {requested}"
            )
        dataset = build_split_dataset(cfg, reader, "validation", logger0, requested)
        self.validation_dataset = dataset
        # Validation is a sharded pass of the shared suite (run_suite builds its
        # own loader); the generic validation loader stays unset.
        shard = ExactShardSampler(len(dataset), self.dist.world_size, self.dist.rank)
        logger0.info(
            f"Yaris validation: {len(dataset)} cases over {self.dist.world_size} ranks "
            f"(this rank: {len(shard)}), {dataset.num_nodes} kept nodes"
        )

    @torch.no_grad()
    def validate(self, epoch):
        """Run the validation pass of ``epoch`` with the configured metrics."""
        if self.validation_dataset is None:
            raise RuntimeError("Validation was not configured")
        was_training = self.model.training
        self.model.eval()
        try:
            result = evaluate_split(
                self.model,
                self.validation_dataset,
                self.data_stats,
                self.cfg,
                self.dist.device,
                world_size=self.dist.world_size,
                rank=self.dist.rank,
                amp=self.amp,
                amp_dtype=self.amp_dtype,
                logger=self.logger0,
            )
            self.last_penetration_summary = result.penetration
            log_summary(self.logger0, epoch + 1, result)
            if self.dist.rank == 0:
                write_outputs(
                    result,
                    self.validation_dataset,
                    self.data_stats,
                    epoch + 1,
                    self.validation_dir,
                )
            return training_stats(result)
        finally:
            self.model.train(was_training)
