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
"""The exact held-out validation suite, shared by the trainer and the evaluators.

One implementation serves :class:`trainers.px_yaris_trainer.YarisFullTrainer` (its
validation epochs), ``evaluate_px_yaris.py`` and ``evaluate_biw.py``: every
held-out case is evaluated exactly once on any rank count
(:class:`ExactShardSampler`), the squared errors are pooled per rollout step
with explicit counts so cases of different horizons mix, and the record carries
the pooled and per-frame vector RMSE in millimetres, the per-case breakdown,
the per-region translational share and node-error percentiles (with a shared
mesh) and the target-relative penetration statistics of
:class:`metrics.penetration_metrics.PenetrationMetrics`.

:func:`run_suite` takes a model, a held-out dataset with the names of its cases,
the frozen normalization statistics and the metric settings and returns a
:class:`SuiteResult`; :func:`metric_fields` turns it into the record's metric
block, which the callers wrap with their own identification keys.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datapipes.datapipe import EPS, simsample_collate
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_original_cwd
from torch.utils.data import DataLoader, Dataset, Sampler

from metrics.penetration_metrics import PenetrationMetrics


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


class NamedCases(Dataset):
    """Serve a dataset's samples with the case name the accumulator reads
    (``yaris_case_name``; the VTP datapipe keeps its records by position)."""

    def __init__(self, dataset, names):
        if len(names) != len(dataset) or len(set(names)) != len(names):
            raise ValueError("one unique name per held-out case is required")
        self.dataset = dataset
        self.names = list(names)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        sample.yaris_case_name = self.names[index]
        return sample


def atomic_json(path, value):
    """Write ``value`` as JSON to ``path`` via a temporary file and atomic replace.

    The result gets the permissions a plain ``open(path, "w")`` would give it
    (the umask applied to 0o666, or the mode of the file it replaces), not the
    0o600 of the temporary file, so a shared statistics directory stays
    readable; a failed dump leaves no temporary file behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        if path.exists():
            mode = path.stat().st_mode & 0o777
        else:
            umask = os.umask(0)
            os.umask(umask)
            mode = 0o666 & ~umask
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


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
    An optional :class:`metrics.penetration_metrics.PenetrationMetrics` accumulates its own statistics, and an
    optional ``node_errors`` tensor ``[N, 2]`` (float64) accumulates per node the
    summed squared physical error (mm^2, all channels) and the frame count.
    Optional ``regions`` = ``(masks [R, N] bool, totals [T, R, 6] float64, cases)``
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


def region_masks(x: torch.Tensor, bounds) -> torch.Tensor:
    """``[R, N]`` bool: the reference-axis regions and the whole mesh last.

    The whole mesh's translational share is the part of the error one rigid
    translation per frame would remove.
    """
    edges = [-float("inf"), *[float(b) for b in bounds], float("inf")]
    return torch.stack(
        [(x >= lo) & (x < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        + [torch.ones_like(x, dtype=torch.bool)]
    )


def region_names(bounds) -> list[str]:
    """Name the x-regions delimited by ``bounds`` plus the pooled ``all``."""
    edges = [-float("inf"), *[float(b) for b in bounds], float("inf")]
    return [f"x_{lo:g}_to_{hi:g}" for lo, hi in zip(edges[:-1], edges[1:])] + ["all"]


def region_summary(masks, totals, per_case, bounds, horizon) -> dict:
    """Translational share of the error per reference-x region (JSON-ready)."""
    totals = totals[:horizon].cpu()
    names = region_names(bounds)
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
            translational_rms_xyz_mm=(t[:, 1:4].sum(0) / frames.sum()).sqrt().tolist(),
            rmse_mm_per_time=(t[:, 4] / frames).sqrt().tolist(),
            translational_rmse_mm_per_time=(t[:, 0] / frames).sqrt().tolist(),
        )
    return out


def node_error_percentiles(node_errors: torch.Tensor) -> tuple[dict, torch.Tensor]:
    """Percentiles of the per-node RMSE (mm) over every validated frame and the
    share of the squared error carried by the worst 1% and 10% of the nodes."""
    rmse = torch.sqrt(node_errors[:, 0] / node_errors[:, 1].clamp_min(1.0))
    values = rmse.double()
    sq = values.square()
    order = torch.argsort(sq, descending=True)
    total = float(sq.sum()) or 1.0
    worst = {}
    for frac in (0.01, 0.10):
        k = max(1, int(frac * values.numel()))
        worst[f"worst_{int(frac * 100)}pct_share"] = float(sq[order[:k]].sum() / total)
    percentiles = dict(
        p50=float(values.quantile(0.5)),
        p90=float(values.quantile(0.9)),
        p99=float(values.quantile(0.99)),
        max=float(values.max()),
        **worst,
    )
    return percentiles, rmse


def data_stats_of(dataset, device) -> dict:
    """The statistics dictionary the models take (as in :class:`train.Trainer`)."""
    return dict(
        node={k: torch.as_tensor(v).to(device) for k, v in dataset.node_stats.items()},
        edge={
            k: torch.as_tensor(v).to(device)
            for k, v in getattr(dataset, "edge_stats", {}).items()
        },
        feature={
            k: torch.as_tensor(v).to(device)
            for k, v in getattr(dataset, "feature_stats", {}).items()
        },
        global_features={
            "mean": dataset.global_stats["global_mean"].to(device),
            "std": dataset.global_stats["global_std"].to(device),
            "keys": list(dataset.global_features_keys or []),
        },
    )


@dataclass
class SuiteResult:
    """What one pass of the suite produced (identical on every rank)."""

    stats: dict  # the pooled tensors of summarize_validation
    per_case: list  # summarize_cases
    cases: list  # sorted held-out case names
    kept_nodes: int
    horizon: int  # number of rollout steps with targets
    penetration: dict | None  # PenetrationMetrics.summary, or None without the probe
    regions: dict | None  # region_summary, or None without node errors
    node_percentiles: dict | None
    node_rmse: torch.Tensor | None  # [N] mm, for the .npy next to the record
    probe: PenetrationMetrics | None
    notes: dict


def run_suite(
    model,
    dataset,
    names,
    data_stats,
    rollout_steps: int,
    device,
    *,
    num_nodes: int,
    world_size: int = 1,
    rank: int = 0,
    amp: bool = False,
    amp_dtype=None,
    penetration_cfg=None,
    region_bounds,
    reference_x: torch.Tensor | None = None,
    node_errors: bool = False,
    facets_present: bool | None = None,
    logger=None,
) -> SuiteResult:
    """Evaluate ``model`` on every case of ``dataset``, sharded over the ranks.

    ``names`` are the case names in the dataset's order (one per sample,
    unique); the record lists them sorted. ``penetration_cfg`` is the
    ``training.validation_penetration`` node (its ``enabled`` key decides).
    Node-error maps and the per-region readout need one shared mesh:
    ``reference_x`` holds the reference x coordinate of every kept node in
    millimetres and ``node_errors`` turns them on; without ``reference_x`` they
    are skipped with a note. ``facets_present`` (when given) skips the
    penetration probe with a note instead of letting it fail on a graph
    without shell facets.
    """
    device = torch.device(device)
    loader = DataLoader(
        NamedCases(dataset, names),
        batch_size=1,
        sampler=ExactShardSampler(len(dataset), world_size, rank),
        collate_fn=simsample_collate,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    notes = {}
    bounds = [float(b) for b in region_bounds]

    probe = None
    if penetration_cfg is not None and bool(penetration_cfg.get("enabled", False)):
        if facets_present is False:
            notes["penetration"] = (
                "no shell facets on the graph (datapipe.contact_surface=false): "
                "penetration metrics skipped"
            )
        else:
            probe = PenetrationMetrics(
                penetration_cfg,
                rollout_steps,
                device,
                logger=logger,
                region_bounds=bounds,
            )

    errors = None
    regions = None
    if node_errors:
        if reference_x is None:
            notes["mesh"] = (
                "held-out cases do not share one mesh: node maps and regions skipped"
            )
        else:
            errors = torch.zeros(
                (int(num_nodes), 2), dtype=torch.float64, device=device
            )
            masks = region_masks(reference_x.to(device), bounds)
            regions = (
                masks,
                torch.zeros(
                    (int(rollout_steps), masks.shape[0], 6),
                    dtype=torch.float64,
                    device=device,
                ),
                [],
            )

    case_records = []
    totals, seen = accumulate_validation(
        model,
        loader,
        data_stats,
        rollout_steps,
        device,
        amp=amp,
        amp_dtype=amp_dtype,
        probe=probe,
        node_errors=errors,
        regions=regions,
        per_case=case_records,
    )
    if probe is not None:
        probe.reduce(world_size)
    if world_size > 1:
        if errors is not None:
            torch.distributed.all_reduce(errors)
            torch.distributed.all_reduce(regions[1])
            gathered_regions = [None] * world_size
            torch.distributed.all_gather_object(gathered_regions, regions[2])
            regions = (
                regions[0],
                regions[1],
                [record for records in gathered_regions for record in records],
            )
        torch.distributed.all_reduce(totals)
        gathered = [None] * world_size
        torch.distributed.all_gather_object(gathered, seen)
        seen = [case for cases in gathered for case in cases]
        gathered_records = [None] * world_size
        torch.distributed.all_gather_object(gathered_records, case_records)
        case_records = [r for records in gathered_records for r in records]
    if sorted(seen) != sorted(names):
        raise RuntimeError("Validation must cover every held-out case exactly once")
    if not torch.isfinite(totals).all():
        raise ValueError("Non-finite validation accumulation")
    stats = summarize_validation(totals)
    horizon = int(stats["nodes_w_time"].numel())
    penetration = None if probe is None else probe.summary(stats["nodes_w_time"])
    percentiles, node_rmse = (None, None)
    if errors is not None:
        percentiles, node_rmse = node_error_percentiles(errors)
    return SuiteResult(
        stats=stats,
        per_case=summarize_cases(case_records),
        cases=sorted(seen),
        kept_nodes=int(num_nodes),
        horizon=horizon,
        penetration=penetration,
        regions=(
            None
            if regions is None
            else region_summary(regions[0], regions[1], regions[2], bounds, horizon)
        ),
        node_percentiles=percentiles,
        node_rmse=node_rmse,
        probe=probe,
        notes=notes,
    )


def metric_fields(result: SuiteResult, pos_std, dt_s: float) -> dict:
    """The record's metric block: pooled errors, time labels, per-case breakdown.

    ``pos_std`` is the normalization scale of the positions (mm); ``dt_s`` the
    frame interval in seconds. The targets start one frame after the observed
    initial one, so the first time label is one interval.
    """
    stats = result.stats
    dt_ms = float(dt_s) * 1000.0
    history = 1
    return dict(
        MSE=stats["MSE"].item(),
        MSE_per_channel=stats["MSE_w_channel"].tolist(),
        MSE_per_time=stats["MSE_w_time"].tolist(),
        vector_RMSE_mm=stats["vector_RMSE_mm"].item(),
        vector_RMSE_mm_per_time=stats["vector_RMSE_mm_w_time"].tolist(),
        target_time_ms=[(t + history) * dt_ms for t in range(result.horizon)],
        nodes_per_time=stats["nodes_w_time"].tolist(),
        per_case=result.per_case,
        pos_std_mm=(torch.as_tensor(pos_std).cpu() + EPS).reshape(-1).tolist(),
    )


def training_stats(result: SuiteResult) -> dict:
    """The float32 tensors the training loop reads from a validation pass."""
    return {
        key: result.stats[key].to(torch.float32)
        for key in ("MSE", "MSE_w_time", "MSE_w_channel", "MSE_w_time_channel")
    }


def log_summary(logger, epoch: int, result: SuiteResult) -> None:
    """Log pooled errors, best and worst cases, the penetration line and the notes."""
    ranked = sorted(result.per_case, key=lambda r: r["vector_RMSE_mm"], reverse=True)
    logger.info(
        f"Validation epoch {epoch}: MSE: {result.stats['MSE'].item():.6f}, vector "
        f"RMSE {result.stats['vector_RMSE_mm'].item():.3f} mm; per-case worst "
        + ", ".join(f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[:3])
        + " | best "
        + ", ".join(f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[-3:])
    )
    if result.penetration is not None:
        logger.info(
            f"Validation epoch {epoch}: penetration "
            + result.probe.log_line(result.penetration)
        )
    for key, note in result.notes.items():
        logger.warning(f"{key}: {note}")


def save_node_rmse(path, node_rmse: torch.Tensor) -> None:
    """Write the per-node RMSE map (mm, float32) next to the record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, node_rmse.cpu().numpy().astype(np.float32))


def require_output_run_dir(output_dir) -> Path:
    """The evaluators' output directory, checked against Hydra's run directory.

    The recipes' ``hydra.run.dir`` is the training run's directory, whose
    ``.hydra`` files an evaluation would overwrite; the evaluators therefore
    require ``hydra.run.dir`` to lie inside ``training.validation_output_dir``
    (for example ``<out>/hydra``). Returns the absolute output directory.
    """
    if not output_dir:
        raise ValueError(
            "pass training.validation_output_dir=<dir>; nothing is written next to "
            "the training run"
        )
    output = Path(str(output_dir))
    if not output.is_absolute():
        output = Path(get_original_cwd()) / output
    output = output.resolve()
    run_dir = Path(HydraConfig.get().runtime.output_dir).resolve()
    if run_dir != output and output not in run_dir.parents:
        raise ValueError(
            f"hydra.run.dir={run_dir} lies outside training.validation_output_dir="
            f"{output}; pass hydra.run.dir={output}/hydra so the evaluation's own "
            ".hydra files are written there and not into the training run"
        )
    return output


def load_evaluation_weights(model, cfg, device, logger, world_size: int = 1) -> int:
    """Load the weights to score: a warm start file, or the training checkpoint.

    ``training.warm_start_model`` names a model-only file (``.mdlus`` or a state
    dict) and scores as epoch 0; otherwise the checkpoint of
    ``training.ckpt_path`` at ``training.load_epoch`` (the latest by default) is
    loaded and its epoch returned.
    """
    from utils.warm_start import load_warm_start

    from physicsnemo.utils import load_checkpoint

    warm_start = cfg.training.get("warm_start_model")
    if warm_start:
        report = load_warm_start(model, warm_start)
        logger.info(f"Weights from {warm_start}: {len(report['loaded'])} tensors")
        return 0
    if world_size > 1:
        torch.distributed.barrier()
    epoch = load_checkpoint(
        cfg.training.ckpt_path,
        models=model,
        epoch=cfg.training.get("load_epoch"),
        device=device,
    )
    if epoch <= 0:
        raise RuntimeError(
            f"No training checkpoint (checkpoint.0.<E>.pt) under "
            f"{cfg.training.ckpt_path}; pass training.warm_start_model=<file>.mdlus "
            "for a weights-only file"
        )
    logger.info(f"Loaded checkpoint {cfg.training.ckpt_path} at epoch {epoch}")
    return int(epoch)


def amp_settings(cfg) -> tuple[bool, torch.dtype]:
    """``(amp, amp_dtype)`` of the recipe, validated as the trainer does."""
    amp_dtypes = {"float16": torch.float16, "bfloat16": torch.bfloat16}
    name = str(cfg.training.get("amp_dtype", "float16"))
    if name not in amp_dtypes:
        raise ValueError(
            f"training.amp_dtype must be 'float16' or 'bfloat16'; got {name!r}"
        )
    return bool(cfg.training.get("amp", True)), amp_dtypes[name]
