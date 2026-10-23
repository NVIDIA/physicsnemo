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


class PenetrationProbe:
    """Crossing-aware contact statistics of predicted and true frames.

    Validation diagnostic: one ``SurfaceContactGraphBuilder`` with the reference
    side test and a reference clearance runs on every frame in physical
    millimetres, so only new contact counts and a node that passed through a
    shell reports its crossing depth. Per node the deepest
    ``relu(-gap - tolerance_mm)`` is kept. ``totals[step, 0 | 1, :]`` accumulate,
    for the prediction (0) and the target (1): crossed pairs, penetrating nodes,
    summed node depth (mm) and the maximum node depth (mm). Any failure disables
    the probe and zeroes its totals: a diagnostic never fails validation.
    """

    COLUMNS = ("crossed_pairs", "penetrating_nodes", "depth_sum_mm", "depth_max_mm")

    def __init__(self, cfg, rollout_steps, device, logger=None):
        from physicsnemo.experimental.models.meshtransolver.surface_contact import (
            SurfaceContactGraphBuilder,
        )

        cfg = dict(cfg or {})
        self.tolerance = float(cfg.get("tolerance_mm", 0.5))
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("validation_penetration.tolerance_mm must be >= 0")
        self.builder = SurfaceContactGraphBuilder(
            activation_distance=float(cfg.get("activation_distance_mm", 5.0)),
            feature_scale=float(cfg.get("feature_scale_mm", 10.0)),
            normal_epsilon=0.1,
            include_velocity=False,
            implementation=str(cfg.get("implementation", "warp")),
            max_pairs=int(cfg.get("max_pairs", 60_000_000)),
            material_fan=True,
            side_from_reference=True,
            reference_clearance=float(cfg.get("reference_clearance_mm", 10.0)),
        )
        self.totals = torch.zeros(
            (int(rollout_steps), 2, 4), dtype=torch.float64, device=device
        )
        self.logger = logger
        self.disabled = False

    def settings(self):
        return dict(
            tolerance_mm=self.tolerance,
            activation_distance_mm=self.builder.activation_distance,
            reference_clearance_mm=self.builder.reference_clearance,
            side_from_reference=True,
        )

    def frame(self, positions, reference, thickness, graph):
        """[4] float64: crossed pairs, penetrating nodes, depth sum, depth max."""
        contact = self.builder(
            positions,
            faces=graph.contact_faces,
            shell_thickness=thickness,
            excluded_pairs=graph.contact_surface_exclusions,
            reference_positions=reference,
        )
        gap = contact.edge_features[:, 4] * self.builder.feature_scale
        depth = (-gap - self.tolerance).clamp_min(0.0)
        node_depth = positions.new_zeros(positions.shape[0]).scatter_reduce(
            0, contact.edge_index[1], depth, reduce="amax", include_self=True
        )
        stats = self.builder.last_discovery_stats or {}
        crossed = positions.new_tensor(float(stats.get("crossed_pairs", 0)))
        return torch.stack(
            (
                crossed,
                (node_depth > 0).sum().to(positions.dtype),
                node_depth.sum(),
                node_depth.max() if node_depth.numel() else positions.new_zeros(()),
            )
        ).double()

    def _disable(self, reason):
        self.disabled = True
        self.totals.zero_()
        if self.logger is not None:
            self.logger.warning(f"Penetration probe disabled: {reason}")

    def accumulate(self, prediction, target, sample, data_stats):
        """Add every step of one case; ``prediction``/``target`` are [N, T, 3]."""
        if self.disabled:
            return
        graph = sample.graph
        if any(
            getattr(graph, name, None) is None
            for name in (
                "contact_faces",
                "contact_surface_exclusions",
                "shell_thickness",
            )
        ):
            self._disable("graph has no contact faces (datapipe.contact_surface=false)")
            return
        try:
            with torch.autocast(device_type=prediction.device.type, enabled=False):
                std = data_stats["node"]["pos_std"].reshape(1, 3).to(prediction.device)
                mean = (
                    data_stats["node"]["pos_mean"].reshape(1, 3).to(prediction.device)
                )
                std, mean = std.float(), mean.float()
                reference = sample.node_features["coords"].float() * std + mean
                thickness = graph.shell_thickness.to(
                    device=prediction.device, dtype=torch.float32
                ).reshape(-1)
                for step in range(int(prediction.shape[1])):
                    for which, frames in ((0, prediction), (1, target)):
                        row = self.frame(
                            frames[:, step].float() * std + mean,
                            reference,
                            thickness,
                            graph,
                        )
                        self.totals[step, which, :3] += row[:3]
                        self.totals[step, which, 3] = torch.maximum(
                            self.totals[step, which, 3], row[3]
                        )
        except Exception as error:  # noqa: BLE001 - diagnostic only
            self._disable(f"{type(error).__name__}: {error}")

    def reduce(self, world_size):
        if world_size > 1:
            sums = self.totals[..., :3].contiguous()
            maxima = self.totals[..., 3].contiguous()
            flag = torch.tensor(
                [1.0 if self.disabled else 0.0], device=self.totals.device
            )
            torch.distributed.all_reduce(sums)
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            self.totals = torch.cat((sums, maxima.unsqueeze(-1)), dim=-1)
            self.disabled = bool(flag.item() > 0)

    def summary(self, nodes_w_time):
        """JSON-ready statistics per step and pooled, for prediction and target."""
        horizon = int(nodes_w_time.numel())
        totals = self.totals[:horizon].cpu()
        nodes = nodes_w_time.cpu().double()
        out = dict(self.settings(), complete=not self.disabled)
        for which, name in ((0, "prediction"), (1, "target")):
            t = totals[:, which]
            out[name] = dict(
                crossed_pairs_per_time=t[:, 0].tolist(),
                penetrating_nodes_per_time=t[:, 1].tolist(),
                mean_depth_mm_per_time=(t[:, 2] / nodes.clamp_min(1)).tolist(),
                max_depth_mm_per_time=t[:, 3].tolist(),
                crossed_pairs=float(t[:, 0].sum()),
                penetrating_node_fraction=float(
                    t[:, 1].sum() / nodes.sum().clamp_min(1)
                ),
                mean_depth_mm=float(t[:, 2].sum() / nodes.sum().clamp_min(1)),
                max_depth_mm=float(t[:, 3].max()) if horizon else 0.0,
            )
        return out

    def log_line(self, summary):
        parts = []
        for name in ("prediction", "target"):
            s = summary[name]
            parts.append(
                f"{name}: crossed {s['crossed_pairs']:.0f}, penetrating "
                f"{100 * s['penetrating_node_fraction']:.3f}% of nodes, mean depth "
                f"{s['mean_depth_mm']:.4f} mm, max {s['max_depth_mm']:.2f} mm"
            )
        return " | ".join(parts) + ("" if summary["complete"] else " (INCOMPLETE)")


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
):
    """Sum squared errors per rollout step over this rank's held-out cases.

    Returns ``totals`` of shape ``[rollout_steps, 5]`` (float64, on ``device``):
    columns 0-2 hold the summed squared normalized error per channel, column 3
    the number of nodes contributing at that step, column 4 the summed squared
    physical error (mm^2, all three channels), plus the list of case names seen.
    An optional :class:`PenetrationProbe` accumulates its own statistics, and an
    optional ``node_errors`` tensor ``[N, 2]`` (float64) accumulates per node the
    summed squared physical error (mm^2, all channels) and the frame count.
    Optional ``regions`` = ``(masks [R, N] bool, totals [T, R, 6] float64)``
    accumulates per region and step the squared norm of the frame-mean error
    (its translational part), that mean's squared x/y/z components, the mean
    squared error and the frame count, all in mm.
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
            extra = getattr(sample, "delta_references", None)
            if extra:
                # Delta-to-reference: average the predictions obtained from
                # every selected training reference (features + baseline).
                keep = (
                    sample.node_features["features"],
                    sample.node_features.get("baseline_all"),
                )
                total = prediction.float()
                try:
                    for paired, displacement in extra:
                        sample.node_features["features"] = paired.to(device)
                        sample.node_features["baseline_all"] = displacement.to(device)
                        more = model(sample=sample, data_stats=data_stats)
                        if hasattr(more, "trajectory"):
                            more = more.trajectory
                        total = total + more.float()
                finally:
                    sample.node_features["features"] = keep[0]
                    if keep[1] is None:
                        sample.node_features.pop("baseline_all", None)
                    else:
                        sample.node_features["baseline_all"] = keep[1]
                prediction = total / (1 + len(extra))
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
        totals[:steps, :3] += torch.square(error).sum(dim=0)
        totals[:steps, 3] += float(target.shape[0])
        totals[:steps, 4] += torch.square(error * pos_std.double()).sum(dim=(0, 2))
        if probe is not None:
            probe.accumulate(prediction, target, sample, data_stats)
        if node_errors is not None:
            node_errors[:, 0] += torch.square(error * pos_std.double()).sum(dim=(1, 2))
            node_errors[:, 1] += float(steps)
        if regions is not None:
            masks, region_totals, per_case = regions
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
            per_case.append({"case": seen[-1], "mean_error_mm": case_means})
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
        out = {"per_case_mean_error_mm": sorted(per_case, key=lambda r: r["case"])}
        for r in range(masks.shape[0]):
            t = totals[:, r]
            frames = t[:, 5].clamp_min(1.0)
            mse = t[:, 4].sum() / frames.sum()
            mean2 = t[:, 0].sum() / frames.sum()
            out[f"x_{bounds[r]:g}_to_{bounds[r + 1]:g}"] = dict(
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
        """Per-node RMSE (mm) over every validated frame, plus node metadata once."""
        import numpy as np

        self.validation_dir.mkdir(parents=True, exist_ok=True)
        rmse = torch.sqrt(node_errors[:, 0] / node_errors[:, 1].clamp_min(1.0))
        np.save(
            self.validation_dir / f"node_rmse_epoch_{epoch + 1:04d}.npy",
            rmse.cpu().numpy().astype(np.float32),
        )
        meta = self.validation_dir / "node_meta.npz"
        if not meta.exists():
            reader = self.validation_dataset.reader
            try:
                store = reader.open(self.validation_dataset.paths[0])
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
                thickness=np.asarray(
                    getattr(
                        self.validation_dataset.graph, "shell_thickness", np.zeros(0)
                    )
                ),
            )

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
            probe = PenetrationProbe(
                self.penetration_probe_cfg,
                self.rollout_steps,
                self.dist.device,
                logger=self.logger0,
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
            masks = torch.stack(
                [(x >= lo) & (x < hi) for lo, hi in zip(bounds[:-1], bounds[1:])]
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
            expected = sorted(path.stem for path in self.validation_dataset.paths)
            if sorted(seen) != expected:
                raise RuntimeError("Validation must cover every CSV case exactly once")
            if not torch.isfinite(totals).all():
                raise ValueError("Non-finite validation accumulation")
            stats = summarize_validation(totals)
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
            if self.dist.rank == 0 and node_errors is not None:
                self._save_node_errors(epoch, node_errors)
            if self.dist.rank == 0:
                reader = self.validation_dataset.reader
                dt_ms = reader.dt * 1000.0
                horizon = int(stats["nodes_w_time"].numel())
                atomic_json(
                    self.validation_dir / f"epoch_{epoch + 1:04d}.json",
                    dict(
                        penetration=penetration,
                        regions=region_summary,
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
                        target_time_ms=[(t + 2) * dt_ms for t in range(horizon)],
                        nodes_per_time=stats["nodes_w_time"].tolist(),
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
