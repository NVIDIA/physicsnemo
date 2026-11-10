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
with explicit counts instead of assuming one horizon.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch
from datapipe import EPS, simsample_collate
from hydra.utils import instantiate
from penetration_metrics import PenetrationMetrics
from torch.utils.data import DataLoader, Sampler
from train import Trainer
from yaris_full import atomic_json


class ExactShardSampler(Sampler):
    """Every case exactly once across ranks; uneven shards and idle ranks allowed."""

    def __init__(self, length, num_replicas, rank):
        if not 0 <= rank < num_replicas:
            raise ValueError("Invalid shard rank")
        self.length, self.replicas, self.rank = (
            int(length),
            int(num_replicas),
            int(rank),
        )

    def __iter__(self):
        return iter(range(self.rank, self.length, self.replicas))

    def __len__(self):
        return len(range(self.rank, self.length, self.replicas))


def accumulate_validation(
    model,
    loader,
    data_stats,
    rollout_steps,
    device,
    amp=False,
    amp_dtype=None,
    probe=None,
    node_errors=None,
    regions=None,
    per_case=None,
):
    """Sum squared errors per rollout step over this rank's held-out cases.

    Returns ``totals`` of shape ``[rollout_steps, 5]`` (float64, on ``device``):
    columns 0-2 hold the summed squared normalized error per channel, column 3
    the number of nodes contributing at that step, column 4 the summed squared
    physical error (mm^2, all three channels), plus the list of case names seen.
    An optional :class:`penetration_metrics.PenetrationMetrics` accumulates its own statistics, and an
    optional ``node_errors`` tensor ``[N, 2]`` (float64) accumulates per node the
    summed squared physical error (mm^2, all channels) and the frame count.
    Optional ``regions`` = ``(masks [R, N] bool, totals [T, R, 6] float64)``
    accumulates per region and step the squared norm of the frame-mean error
    (its translational part), that mean's squared x/y/z components, the mean
    squared error and the frame count, all in mm. Optional ``per_case`` (a list)
    receives one record per held-out case: its name, kept-node count and the
    summed squared physical error per step (mm^2), for the per-case breakdown.
    """
    totals = torch.zeros((int(rollout_steps), 5), dtype=torch.float64, device=device)
    pos_std = data_stats["node"]["pos_std"].reshape(1, 1, 3).to(device)
    seen = []
    device_type = "cuda" if torch.device(device).type == "cuda" else "cpu"
    for batch in loader:
        sample = batch[0].to(device)
        seen.append(getattr(sample, "yaris_case_name", str(len(seen))))
        with torch.autocast(
            device_type=device_type,
            enabled=bool(amp),
            dtype=amp_dtype if amp_dtype is not None else torch.bfloat16,
        ):
            prediction = model(sample=sample, data_stats=data_stats)
            if hasattr(prediction, "trajectory"):
                prediction = prediction.trajectory
        target = sample.node_target
        if prediction.shape != target.shape or target.shape[-1] != 3:
            raise ValueError(
                f"Expected displacement-only trajectory {tuple(target.shape)}, got "
                f"{tuple(prediction.shape)}"
            )
        steps = int(target.shape[1])
        if steps > totals.shape[0]:
            raise ValueError(
                f"Case has {steps} target steps but rollout_steps={totals.shape[0]}"
            )
        error = (prediction.float() - target.float()).double()  # [N, T, 3]
        physical_sq = torch.square(error * pos_std.double()).sum(dim=(0, 2))  # [T]
        totals[:steps, :3] += torch.square(error).sum(dim=0)
        totals[:steps, 3] += float(target.shape[0])
        totals[:steps, 4] += physical_sq
        if per_case is not None:
            per_case.append(
                {
                    "case": seen[-1],
                    "nodes": int(target.shape[0]),
                    "sum_sq_mm_per_time": physical_sq.cpu().tolist(),
                }
            )
        if probe is not None:
            probe.accumulate(prediction, target, sample, data_stats)
        if node_errors is not None:
            node_errors[:, 0] += torch.square(error * pos_std.double()).sum(dim=(1, 2))
            node_errors[:, 1] += float(steps)
        if regions is not None:
            masks, region_totals, region_cases = regions
            physical = error * pos_std.double()  # [N, T, 3]
            case_means = []
            for r in range(masks.shape[0]):
                part = physical[masks[r]]  # [n, T, 3]
                if part.shape[0] == 0:
                    case_means.append([])
                    continue
                mean = part.mean(dim=0)  # [T, 3]
                region_totals[:steps, r, 0] += mean.square().sum(-1)
                region_totals[:steps, r, 1:4] += mean.square()
                region_totals[:steps, r, 4] += part.square().sum(-1).mean(dim=0)
                region_totals[:steps, r, 5] += 1.0
                case_means.append(mean.cpu().tolist())
            region_cases.append({"case": seen[-1], "mean_error_mm": case_means})
    return totals, seen


def summarize_validation(totals):
    """Pool per-step sums into the crash validation metrics (normalized units)."""
    counts = totals[:, 3]
    active = torch.nonzero(counts > 0).flatten()
    if active.numel() == 0:
        raise ValueError("No validation targets were accumulated")
    horizon = int(active[-1]) + 1
    totals = totals[:horizon]
    if (totals[:, 3] <= 0).any():
        raise ValueError("Unsupported hole in validation time coverage")
    counts = totals[:, 3:4]
    time_channel = totals[:, :3] / counts  # [T, 3]
    channel = totals[:, :3].sum(0) / totals[:, 3].sum()  # [3]
    mse = channel.mean()
    physical_rmse_time = torch.sqrt(totals[:, 4] / totals[:, 3])
    physical_rmse = torch.sqrt(totals[:, 4].sum() / totals[:, 3].sum())
    return dict(
        MSE=mse,
        MSE_w_time=time_channel.mean(1),
        MSE_w_channel=channel,
        MSE_w_time_channel=time_channel,
        vector_RMSE_mm=physical_rmse,
        vector_RMSE_mm_w_time=physical_rmse_time,
        nodes_w_time=totals[:, 3],
    )


def summarize_cases(records):
    """Per-case vector RMSE (mm), pooled over each case's steps, sorted by name."""
    summary = []
    for record in sorted(records, key=lambda r: r["case"]):
        sums = torch.as_tensor(record["sum_sq_mm_per_time"], dtype=torch.float64)
        nodes = float(record["nodes"])
        summary.append(
            dict(
                case=record["case"],
                steps=int(sums.numel()),
                vector_RMSE_mm=float(torch.sqrt(sums.sum() / (nodes * sums.numel()))),
                vector_RMSE_mm_per_time=torch.sqrt(sums / nodes).tolist(),
            )
        )
    return summary


class YarisFullTrainer(Trainer):
    """Unchanged training loop; exact, horizon-aware held-out validation."""

    def __init__(self, cfg, logger0):
        if not math.isclose(
            float(cfg.model.dt),
            float(cfg.reader.raw_dt) * int(cfg.reader.stride),
            abs_tol=1e-12,
        ):
            raise ValueError("Model dt differs from raw frame interval times stride")
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
        self.penetration_probe_cfg = cfg.training.get("validation_penetration")
        self.validation_node_errors = bool(
            cfg.training.get("validation_node_errors", False)
        )
        # Reference-x bounds splitting the mesh into regions (rear, ..., front)
        # for the translational-error breakdown written with the node errors.
        self.validation_region_bounds = [
            float(v)
            for v in cfg.training.get("validation_region_bounds", [-2000.0, -1000.0])
        ]
        if requested <= 0:
            return
        reader = instantiate(cfg.reader)
        names = reader.splits["validation"]
        if requested != len(names):
            raise ValueError(
                f"training.num_validation_samples must equal the {len(names)} CSV "
                f"validation cases, got {requested}"
            )
        dataset = instantiate(
            cfg.datapipe,
            name="crash_validation",
            reader=reader,
            split="validation",
            logger=logger0,
            stats_mode="load",
            sample_type="all_time_steps",
            data_dir=cfg.training.raw_data_dir_validation,
            num_samples=requested,
        )
        sampler = ExactShardSampler(len(dataset), self.dist.world_size, self.dist.rank)
        self.val_dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=simsample_collate,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        self.validation_dataset = dataset
        logger0.info(
            f"Yaris validation: {len(dataset)} cases over {self.dist.world_size} ranks "
            f"(this rank: {len(sampler)}), {dataset.num_nodes} kept nodes"
        )

    def _region_summary(self, regions, horizon):
        """Translational share of the error per reference-x region (JSON-ready)."""
        masks, totals, per_case = regions
        totals = totals[:horizon].cpu()
        bounds = [-float("inf"), *self.validation_region_bounds, float("inf")]
        names = [f"x_{lo:g}_to_{hi:g}" for lo, hi in zip(bounds[:-1], bounds[1:])] + [
            "all"
        ]
        out = {"per_case_mean_error_mm": sorted(per_case, key=lambda r: r["case"])}
        for r in range(masks.shape[0]):
            t = totals[:, r]
            frames = t[:, 5].clamp_min(1.0)
            mse = t[:, 4].sum() / frames.sum()
            mean2 = t[:, 0].sum() / frames.sum()
            out[names[r]] = dict(
                nodes=int(masks[r].sum()),
                rmse_mm=float(mse.sqrt()),
                translational_rmse_mm=float(mean2.sqrt()),
                translational_share=float(mean2 / mse) if mse > 0 else 0.0,
                translational_rms_xyz_mm=(t[:, 1:4].sum(0) / frames.sum())
                .sqrt()
                .tolist(),
                rmse_mm_per_time=(t[:, 4] / frames).sqrt().tolist(),
                translational_rmse_mm_per_time=(t[:, 0] / frames).sqrt().tolist(),
            )
        return out

    def _save_node_errors(self, epoch, node_errors):
        """Per-node RMSE (mm) over every validated frame, plus node metadata once.
        Returns the percentile summary of the node RMSE for the validation JSON."""
        import numpy as np

        self.validation_dir.mkdir(parents=True, exist_ok=True)
        rmse = torch.sqrt(node_errors[:, 0] / node_errors[:, 1].clamp_min(1.0))
        np.save(
            self.validation_dir / f"node_rmse_epoch_{epoch + 1:04d}.npy",
            rmse.cpu().numpy().astype(np.float32),
        )
        values = rmse.double()
        sq = values.square()
        order = torch.argsort(sq, descending=True)
        total = float(sq.sum()) or 1.0
        worst = {}
        for frac in (0.01, 0.10):
            k = max(1, int(frac * values.numel()))
            worst[f"worst_{int(frac * 100)}pct_share"] = float(
                sq[order[:k]].sum() / total
            )
        percentiles = dict(
            p50=float(values.quantile(0.5)),
            p90=float(values.quantile(0.9)),
            p99=float(values.quantile(0.99)),
            max=float(values.max()),
            **worst,
        )
        meta = self.validation_dir / "node_meta.npz"
        if not meta.exists():
            reader = self.validation_dataset.reader
            store = reader.open(self.validation_dataset.paths[0])
            try:
                part_id = np.asarray(store["nodes/part_id"][:]).reshape(-1)[
                    reader.keep_ids
                ]
            except Exception:  # noqa: BLE001 - metadata is optional
                part_id = np.full(len(reader.keep_ids), -1, dtype=np.int64)
            np.savez(
                meta,
                part_id=part_id,
                reference=np.asarray(reader.reference, dtype=np.float32),
                keep_ids=np.asarray(reader.keep_ids, dtype=np.int64),
                # the first validation design's thickness (it varies per design)
                thickness=np.asarray(reader.thickness(store), dtype=np.float32),
            )
        return percentiles

    @torch.no_grad()
    def validate(self, epoch):
        """Run the validation pass of ``epoch`` with the configured metrics."""
        if self.val_dataloader is None:
            raise RuntimeError("Validation was not configured")
        was_training = self.model.training
        self.model.eval()
        probe = None
        if self.penetration_probe_cfg is not None and bool(
            self.penetration_probe_cfg.get("enabled", False)
        ):
            probe = PenetrationMetrics(
                self.penetration_probe_cfg,
                self.rollout_steps,
                self.dist.device,
                logger=self.logger0,
                region_bounds=self.validation_region_bounds,
            )
        node_errors = None
        regions = None
        if self.validation_node_errors:
            node_errors = torch.zeros(
                (int(self.validation_dataset.num_nodes), 2),
                dtype=torch.float64,
                device=self.dist.device,
            )
            x = torch.as_tensor(
                self.validation_dataset.reader.reference[:, 0], device=self.dist.device
            )
            bounds = [-float("inf"), *self.validation_region_bounds, float("inf")]
            # The x regions plus the whole mesh last: its translational share is
            # the part of the error one rigid translation per frame would remove.
            masks = torch.stack(
                [(x >= lo) & (x < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
                + [torch.ones_like(x, dtype=torch.bool)]
            )
            regions = (
                masks,
                torch.zeros(
                    (int(self.rollout_steps), masks.shape[0], 6),
                    dtype=torch.float64,
                    device=self.dist.device,
                ),
                [],
            )
        case_records = []
        try:
            totals, seen = accumulate_validation(
                self.model,
                self.val_dataloader,
                self.data_stats,
                self.rollout_steps,
                self.dist.device,
                amp=self.amp,
                amp_dtype=self.amp_dtype,
                probe=probe,
                node_errors=node_errors,
                regions=regions,
                per_case=case_records,
            )
            if probe is not None:
                probe.reduce(self.dist.world_size)
            if node_errors is not None and self.dist.world_size > 1:
                torch.distributed.all_reduce(node_errors)
                torch.distributed.all_reduce(regions[1])
                gathered_cases = [None] * self.dist.world_size
                torch.distributed.all_gather_object(gathered_cases, regions[2])
                regions = (
                    regions[0],
                    regions[1],
                    [record for records in gathered_cases for record in records],
                )
            if self.dist.world_size > 1:
                torch.distributed.all_reduce(totals)
                gathered = [None] * self.dist.world_size
                torch.distributed.all_gather_object(gathered, seen)
                seen = [case for cases in gathered for case in cases]
                gathered_records = [None] * self.dist.world_size
                torch.distributed.all_gather_object(gathered_records, case_records)
                case_records = [r for records in gathered_records for r in records]
            expected = sorted(path.stem for path in self.validation_dataset.paths)
            if sorted(seen) != expected:
                raise RuntimeError("Validation must cover every CSV case exactly once")
            if not torch.isfinite(totals).all():
                raise ValueError("Non-finite validation accumulation")
            stats = summarize_validation(totals)
            per_case = summarize_cases(case_records)
            ranked = sorted(per_case, key=lambda r: r["vector_RMSE_mm"], reverse=True)
            self.logger0.info(
                f"Validation epoch {epoch + 1}: per-case RMSE mm worst "
                + ", ".join(
                    f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[:3]
                )
                + " | best "
                + ", ".join(
                    f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[-3:]
                )
            )
            penetration = (
                None if probe is None else probe.summary(stats["nodes_w_time"])
            )
            if penetration is not None:
                self.logger0.info(
                    f"Validation epoch {epoch + 1}: penetration "
                    + probe.log_line(penetration)
                )
            region_summary = (
                None
                if regions is None
                else self._region_summary(regions, stats["nodes_w_time"].numel())
            )
            node_percentiles = None
            if self.dist.rank == 0 and node_errors is not None:
                node_percentiles = self._save_node_errors(epoch, node_errors)
            if self.dist.rank == 0:
                reader = self.validation_dataset.reader
                dt_ms = reader.dt * 1000.0
                # The targets start one frame after the observed initial one.
                history = 1
                horizon = int(stats["nodes_w_time"].numel())
                atomic_json(
                    self.validation_dir / f"epoch_{epoch + 1:04d}.json",
                    dict(
                        penetration=penetration,
                        regions=region_summary,
                        node_error_percentiles=node_percentiles,
                        epoch=epoch + 1,
                        cases=sorted(seen),
                        kept_nodes=int(self.validation_dataset.num_nodes),
                        excluded_nodes=int(reader.total_nodes - len(reader.keep_ids)),
                        metric=(
                            "normalized position MSE pooled over every held-out "
                            "(case, step, node, channel) entry; outlier nodes excluded"
                        ),
                        MSE=stats["MSE"].item(),
                        MSE_per_channel=stats["MSE_w_channel"].tolist(),
                        MSE_per_time=stats["MSE_w_time"].tolist(),
                        vector_RMSE_mm=stats["vector_RMSE_mm"].item(),
                        vector_RMSE_mm_per_time=stats["vector_RMSE_mm_w_time"].tolist(),
                        target_time_ms=[(t + history) * dt_ms for t in range(horizon)],
                        nodes_per_time=stats["nodes_w_time"].tolist(),
                        per_case=per_case,
                        pos_std_mm=(self.data_stats["node"]["pos_std"] + EPS)
                        .reshape(-1)
                        .tolist(),
                    ),
                )
            return {
                key: stats[key].to(torch.float32)
                for key in ("MSE", "MSE_w_time", "MSE_w_channel", "MSE_w_time_channel")
            }
        finally:
            self.model.train(was_training)
