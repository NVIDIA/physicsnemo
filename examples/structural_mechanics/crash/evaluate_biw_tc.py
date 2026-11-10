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
"""Score a body-in-white (VTP, structural-graph datapipe) time-conditional checkpoint with
the evaluation suite of the Yaris recipes.

The body-in-white recipes train with the generic :class:`train.Trainer`, whose validation
is the pooled normalized MSE only. This script applies the suite that
:class:`yaris_full_trainer.YarisFullTrainer` wires into its validation to a body-in-white
checkpoint: the exact held-out validation (every case once on any rank count)
with the pooled and per-time vector RMSE in millimetres, per-case and
per-region breakdowns, node-error maps and percentiles, the target-relative
penetration metrics (:class:`penetration_metrics.PenetrationMetrics`: shell
surface and, with ``datapipe.rigid_wall_position``, the rigid wall) and the
temporal-smoothness and per-part kinematics readout
(:class:`kinematics_readout.KinematicsReadout`). Runs inside the training
environment, sharded over GPUs like training::

    torchrun --standalone --nproc_per_node=4 evaluate_biw_tc.py \\
        --config-name crash_deformer_knn_time_conditional \\
        training.raw_data_dir=<train> training.raw_data_dir_validation=<validation> \\
        reader.cache_dir=<vtp cache with cells> ++reader.include_contact_topology=true \\
        ++datapipe.local_knn_cache_dir=<knn cache> datapipe.stats_dir=<run>/stats \\
        ++datapipe.contact_surface=true ++datapipe.contact_surface_exclusion=one_ring \\
        ++datapipe.rigid_wall_position=-29.8 \\
        training.ckpt_path=<run>/checkpoints +training.load_epoch=100 \\
        +training.validation_output_dir=<out> hydra.run.dir=<out>

Only the validation split is built; the normalization statistics are loaded
from ``datapipe.stats_dir`` (the run's ``stats`` directory, written when the
training split was built), so nothing is ever written next to the run. The
record goes to ``<out>/epoch_<E>.json`` with the keys of the Yaris validation
JSON, plus ``node_rmse_epoch_<E>.npy`` and ``kinematics_epoch_<E>.npz``.

Options (all under ``training``): ``validation_penetration`` (the metric
settings; ``enabled: false`` skips it), ``validation_region_bounds`` (reference
``x`` bounds of the regions, default ``[-3000, -1500]``: the body-in-white mesh spans
``x`` from -4045 mm at the rear to -39 mm at the front and crashes into a wall
at ``x = -29.8``), ``validation_node_errors`` and ``validation_kinematics``
(default on). The parts of the kinematics readout are the connected
components of the structural edges. Metrics whose inputs are missing are
reported as ``null`` with a note: the penetration metrics without shell facets
(``datapipe.contact_surface=false``), the wall family without a wall, and the
node maps, regions and kinematics when the held-out cases do not share one
mesh. Model options such as ``model.penetration_projection_iterations=4
model.penetration_projection_damping=0.5 model.clamp_to_wall=true`` evaluate a
checkpoint with the training-free post-processing; never pass ``model.contact_*``
overrides when scoring a contact model (they change its searcher).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import numpy as np
import torch
from datapipe import EPS, simsample_collate
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from penetration_metrics import PenetrationMetrics
from torch.utils.data import DataLoader, Dataset
from yaris_full_trainer import (
    ExactShardSampler,
    accumulate_validation,
    summarize_cases,
    summarize_validation,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Reference-x regions of the body-in-white mesh: rear (x < -3000), middle, front (x >= -1500).
DEFAULT_REGION_BOUNDS = (-3000.0, -1500.0)


class NamedCases(Dataset):
    """Serve a dataset's samples with the case name the shared accumulator reads
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


def case_names(data_dir, count: int) -> list[str]:
    """Stems of the split's VTP files in the reader's order, or indices."""
    try:
        from vtp_reader import find_run_folders

        names = [Path(path).stem for path in find_run_folders(str(data_dir))[:count]]
    except Exception:  # noqa: BLE001 - names are labels only
        names = []
    return names + [str(index) for index in range(len(names), count)]


def atomic_json(path, payload) -> None:
    """Write ``payload`` as JSON through a temporary file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    with open(temporary, "w") as handle:
        json.dump(payload, handle, indent=1)
    os.replace(temporary, path)


def build_validation(cfg, logger0):
    """The held-out dataset of a body-in-white recipe with the run's statistics loaded."""
    count = int(cfg.training.num_validation_samples)
    if count <= 0:
        raise ValueError("training.num_validation_samples must be positive")
    reader = instantiate(cfg.reader)
    return instantiate(
        cfg.datapipe,
        name="crash_validation",
        reader=reader,
        split="validation",
        logger=logger0,
        stats_mode="load",
        sample_type="all_time_steps",
        data_dir=cfg.training.raw_data_dir_validation,
        num_samples=count,
    )


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


def shared_mesh(dataset) -> bool:
    """Whether every held-out case has the first case's node count and edges."""
    first = dataset.graphs[0]
    for graph in dataset.graphs[1:]:
        if int(graph.num_nodes) != int(first.num_nodes):
            return False
        if graph.edge_index is not first.edge_index and not torch.equal(
            graph.edge_index, first.edge_index
        ):
            return False
    return True


def reference_mm(dataset, index: int = 0) -> torch.Tensor:
    """Initial positions of one case in millimetres, float64 ``[N, 3]``."""
    std = torch.as_tensor(dataset.node_stats["pos_std"]).double().reshape(1, 3)
    mean = torch.as_tensor(dataset.node_stats["pos_mean"]).double().reshape(1, 3)
    return dataset.mesh_pos_seq[index][0].double() * (std + EPS) + mean


def part_structure(dataset):
    """Part index ``[N]``, labels and the grouping of the kinematics readout:
    the connected components of the structural edges (the body-in-white VTP files carry no
    part ids)."""
    graph = dataset.graphs[0]
    part_index = graph.component_id.reshape(-1).long()
    count = int(part_index.max()) + 1
    return part_index, np.arange(count), "connected components of the structural edges"


def region_masks(x: torch.Tensor, bounds) -> torch.Tensor:
    """``[R, N]`` bool: the reference-axis regions and the whole mesh last."""
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
    """Translational share of the error per reference-x region (JSON-ready), as
    :meth:`yaris_full_trainer.YarisFullTrainer._region_summary`."""
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


def run_suite(
    model,
    dataset,
    names,
    data_stats,
    rollout_steps: int,
    device,
    *,
    world_size: int = 1,
    rank: int = 0,
    amp: bool = False,
    amp_dtype=None,
    penetration_cfg=None,
    region_bounds=DEFAULT_REGION_BOUNDS,
    node_errors: bool = True,
    kinematics: bool = True,
    logger=None,
) -> dict:
    """Evaluate ``model`` on every case of ``dataset`` (sharded over the ranks).

    Returns the JSON-ready record of the suite (identical on every rank after
    the reductions) and the accumulators behind it (``node_rmse`` ``[N]`` in mm
    or ``None``, the kinematics readout and the penetration probe or ``None``)
    for the files written next to the record."""
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
    shared = shared_mesh(dataset)
    if not shared:
        notes["mesh"] = (
            "held-out cases do not share one mesh: node maps, regions and "
            "kinematics skipped"
        )
    first = dataset.graphs[0]
    bounds = [float(b) for b in region_bounds]

    probe = None
    penetration_enabled = penetration_cfg is None or bool(
        dict(penetration_cfg).get("enabled", True)
    )
    if penetration_enabled:
        if any(
            getattr(first, name, None) is None
            for name in (
                "contact_faces",
                "contact_surface_exclusions",
                "shell_thickness",
            )
        ):
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
            if (
                getattr(first, "rigid_wall", None) is None
                and dict(penetration_cfg or {}).get("wall_position_mm") is None
            ):
                notes["wall"] = (
                    "no rigid wall (datapipe.rigid_wall_position unset): wall "
                    "penetration statistics skipped"
                )

    readout = None
    part_source = None
    if kinematics and shared:
        from kinematics_readout import KinematicsReadout

        part_index, part_ids, part_source = part_structure(dataset)
        readout = KinematicsReadout(
            part_index,
            part_ids,
            reference_mm(dataset),
            float(getattr(dataset, "dt", 5e-3)),
            bounds,
            int(rollout_steps),
            device,
            logger=logger,
        )

    errors = None
    regions = None
    if node_errors and shared:
        n = int(first.num_nodes)
        errors = torch.zeros((n, 2), dtype=torch.float64, device=device)
        x = reference_mm(dataset)[:, 0].to(device)
        masks = region_masks(x, bounds)
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
        kinematics=readout,
    )
    if probe is not None:
        probe.reduce(world_size)
    if readout is not None:
        readout.reduce(world_size)
    if world_size > 1:
        torch.distributed.all_reduce(totals)
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
    per_case = summarize_cases(case_records)
    penetration = None if probe is None else probe.summary(stats["nodes_w_time"])
    kinematics_summary = None if readout is None else readout.summary()
    percentiles, node_rmse = (None, None)
    if errors is not None:
        percentiles, node_rmse = node_error_percentiles(errors)
    dt_ms = float(getattr(dataset, "dt", 5e-3)) * 1000.0
    history = 1  # the targets start one frame after the observed initial one
    record = dict(
        penetration=penetration,
        regions=(
            None
            if regions is None
            else region_summary(regions[0], regions[1], regions[2], bounds, horizon)
        ),
        kinematics=kinematics_summary,
        node_error_percentiles=percentiles,
        notes=notes,
        cases=sorted(seen),
        kept_nodes=int(first.num_nodes),
        mesh_shared=shared,
        part_grouping=part_source,
        region_bounds_mm=bounds,
        dt_s=dt_ms / 1000.0,
        metric=(
            "normalized position MSE pooled over every held-out "
            "(case, step, node, channel) entry"
        ),
        MSE=stats["MSE"].item(),
        MSE_per_channel=stats["MSE_w_channel"].tolist(),
        MSE_per_time=stats["MSE_w_time"].tolist(),
        vector_RMSE_mm=stats["vector_RMSE_mm"].item(),
        vector_RMSE_mm_per_time=stats["vector_RMSE_mm_w_time"].tolist(),
        target_time_ms=[(t + history) * dt_ms for t in range(horizon)],
        nodes_per_time=stats["nodes_w_time"].tolist(),
        per_case=per_case,
        pos_std_mm=(torch.as_tensor(data_stats["node"]["pos_std"]).cpu() + EPS)
        .reshape(-1)
        .tolist(),
    )
    return record, dict(node_rmse=node_rmse, kinematics=readout, probe=probe)


def log_summary(logger, epoch, record, extras) -> None:
    """Log pooled errors, best and worst cases and the penetration summary."""
    ranked = sorted(record["per_case"], key=lambda r: r["vector_RMSE_mm"], reverse=True)
    logger.info(
        f"Validation epoch {epoch}: MSE: {record['MSE']:.6f}, vector RMSE "
        f"{record['vector_RMSE_mm']:.3f} mm; per-case worst "
        + ", ".join(f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[:3])
        + " | best "
        + ", ".join(f"{r['case']} {r['vector_RMSE_mm']:.3f}" for r in ranked[-3:])
    )
    if record["penetration"] is not None:
        logger.info(
            f"Validation epoch {epoch}: penetration "
            + extras["probe"].log_line(record["penetration"])
        )
    if record["kinematics"] is not None:
        logger.info(
            f"Validation epoch {epoch}: kinematics "
            + extras["kinematics"].log_line(record["kinematics"])
        )
    for key, note in record["notes"].items():
        logger.warning(f"{key}: {note}")


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="crash_deformer_knn_time_conditional",
)
def main(cfg: DictConfig) -> None:
    """Score a body-in-white checkpoint and write the validation record."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(PythonLogger("evaluate"), dist)
    output_dir = cfg.training.get("validation_output_dir")
    if not output_dir:
        raise ValueError(
            "pass +training.validation_output_dir=<dir>; nothing is written "
            "next to the run"
        )
    output_dir = Path(output_dir)
    with open_dict(cfg):
        if "stats_dir" not in cfg.datapipe or cfg.datapipe.stats_dir in (None, "stats"):
            # Hydra's run dir is the output dir here; the run's statistics live
            # next to its checkpoints.
            cfg.datapipe.stats_dir = str(Path(cfg.training.ckpt_path).parent / "stats")
    stats_dir = Path(cfg.datapipe.stats_dir)
    if not (stats_dir / "node_stats.json").exists():
        raise FileNotFoundError(
            f"{stats_dir} has no node_stats.json; point datapipe.stats_dir at the "
            "run's stats directory"
        )
    dataset = build_validation(cfg, logger0)
    names = case_names(cfg.training.raw_data_dir_validation, len(dataset))
    data_stats = data_stats_of(dataset, dist.device)
    rollout_steps = int(cfg.training.num_time_steps) - 1

    model = instantiate(cfg.model).to(dist.device)
    warm_start = cfg.training.get("warm_start_model")
    epoch = 0
    if warm_start:
        from utils import load_warm_start

        load_warm_start(model, warm_start, allowed_missing_prefixes=())
        logger0.info(f"Weights from {warm_start}")
    else:
        if dist.world_size > 1:
            torch.distributed.barrier()
        epoch = load_checkpoint(
            cfg.training.ckpt_path,
            models=model,
            epoch=cfg.training.get("load_epoch"),
            device=dist.device,
        )
        if epoch <= 0:
            raise RuntimeError(f"No checkpoint found under {cfg.training.ckpt_path}")
    model.eval()
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[
        cfg.training.get("amp_dtype", "bfloat16")
    ]
    penetration_cfg = cfg.training.get("validation_penetration")
    penetration_cfg = (
        {"enabled": True}
        if penetration_cfg is None
        else OmegaConf.to_container(penetration_cfg, resolve=True)
    )
    bounds = cfg.training.get("validation_region_bounds")
    bounds = DEFAULT_REGION_BOUNDS if bounds is None else [float(v) for v in bounds]
    logger0.info(
        f"Evaluating epoch {epoch} of {cfg.training.ckpt_path} on {dist.world_size} "
        f"rank(s): {len(dataset)} held-out cases, regions {region_names(bounds)}; "
        f"records under {output_dir}"
    )
    with torch.no_grad():
        record, extras = run_suite(
            model,
            dataset,
            names,
            data_stats,
            rollout_steps,
            dist.device,
            world_size=dist.world_size,
            rank=dist.rank,
            amp=bool(cfg.training.get("amp", True)),
            amp_dtype=amp_dtype,
            penetration_cfg=penetration_cfg,
            region_bounds=bounds,
            node_errors=bool(cfg.training.get("validation_node_errors", True)),
            kinematics=bool(cfg.training.get("validation_kinematics", True)),
            logger=logger0,
        )
    record["epoch"] = int(epoch)
    record["checkpoint"] = str(cfg.training.ckpt_path)
    record["config"] = OmegaConf.to_container(cfg.model, resolve=True)
    log_summary(logger0, epoch, record, extras)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / f"epoch_{epoch:04d}.json", record)
        if extras["node_rmse"] is not None:
            np.save(
                output_dir / f"node_rmse_epoch_{epoch:04d}.npy",
                extras["node_rmse"].cpu().numpy().astype(np.float32),
            )
        if extras["kinematics"] is not None:
            extras["kinematics"].save(output_dir / f"kinematics_epoch_{epoch:04d}.npz")
        logger0.info(f"Record written to {output_dir / f'epoch_{epoch:04d}.json'}")


if __name__ == "__main__":
    main()
