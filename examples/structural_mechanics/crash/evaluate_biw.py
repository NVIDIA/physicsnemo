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
the evaluation suite of the PX Yaris recipes.

The body-in-white recipes train with the generic :class:`train.Trainer`, whose validation
is the pooled normalized MSE only. This script applies the suite that
:class:`trainers.px_yaris_trainer.YarisFullTrainer` wires into its validation to a body-in-white
checkpoint: the exact held-out validation (every case once on any rank count)
with the pooled and per-time vector RMSE in millimetres, per-case and
per-region breakdowns, node-error maps and percentiles, the target-relative
penetration metrics (:class:`metrics.penetration_metrics.PenetrationMetrics`: shell
surface and, with ``datapipe.rigid_wall_position``, the rigid wall). Runs
inside the training
environment, sharded over GPUs like training::

    torchrun --standalone --nproc_per_node=4 evaluate_biw.py \\
        --config-name biw_yaris_deformer \\
        training.raw_data_dir=<train> training.raw_data_dir_validation=<validation> \\
        reader.cache_dir=<vtp cache with cells> reader.include_contact_topology=true \\
        datapipe.local_knn_cache_dir=<knn cache> datapipe.stats_dir=<run>/stats \\
        datapipe.contact_surface=true datapipe.contact_surface_exclusion=one_ring \\
        datapipe.rigid_wall_position=-29.8 \\
        training.ckpt_path=<run>/checkpoints training.load_epoch=100 \\
        training.validation_output_dir=<out> hydra.run.dir=<out>

Only the validation split is built; the normalization statistics are loaded
from ``datapipe.stats_dir`` (the run's ``stats`` directory, written when the
training split was built), so nothing is ever written next to the run, and
``hydra.run.dir`` must lie inside ``training.validation_output_dir``. The suite
is :mod:`metrics.validation_suite`, shared with the Yaris trainer and evaluator. The
record goes to ``<out>/epoch_<E>.json`` with the keys of the PX Yaris validation
JSON, plus ``node_rmse_epoch_<E>.npy``.

Options (all under ``training``): ``validation_penetration`` (the metric
settings; ``enabled: false`` skips it), ``validation_region_bounds`` (reference
``x`` bounds of the regions, default ``[-3000, -1500]``: the body-in-white mesh spans
``x`` from -4045 mm at the rear to -39 mm at the front and crashes into a wall
at ``x = -29.8``) and ``validation_node_errors`` (default on). Metrics whose
inputs are missing are reported as ``null`` with a note: the penetration
metrics without shell facets (``datapipe.contact_surface=false``), the wall
family without a wall, and the node maps and regions when the held-out cases
do not share one mesh. ``model.clamp_to_wall=true`` evaluates a checkpoint with
the rigid-wall clamp; never pass ``model.contact_*`` overrides when scoring a
contact model (they change its searcher).
"""

from __future__ import annotations

from pathlib import Path

import hydra
import torch
from datapipes.datapipe import EPS
from hydra.utils import instantiate
from metrics.validation_suite import (
    amp_settings,
    atomic_json,
    data_stats_of,
    load_evaluation_weights,
    log_summary,
    metric_fields,
    region_names,
    require_output_run_dir,
    run_suite,
    save_node_rmse,
)
from omegaconf import DictConfig, OmegaConf, open_dict

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

# Reference-x regions of the body-in-white mesh: rear (x < -3000), middle, front (x >= -1500).
DEFAULT_REGION_BOUNDS = (-3000.0, -1500.0)


def case_names(data_dir, count: int) -> list[str]:
    """Stems of the split's VTP files in the reader's order, or indices."""
    try:
        from readers.vtp_reader import find_run_folders

        names = [Path(path).stem for path in find_run_folders(str(data_dir))[:count]]
    except Exception:  # noqa: BLE001 - names are labels only
        names = []
    return names + [str(index) for index in range(len(names), count)]


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


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="biw_yaris_deformer",
)
def main(cfg: DictConfig) -> None:
    """Score a body-in-white checkpoint and write the validation record."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(PythonLogger("evaluate"), dist)
    output_dir = require_output_run_dir(cfg.training.get("validation_output_dir"))
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

    model = instantiate(cfg.model)
    epoch = load_evaluation_weights(
        model, cfg, dist.device, logger0, world_size=dist.world_size
    )
    model.to(dist.device).eval()
    amp, amp_dtype = amp_settings(cfg)
    penetration_cfg = cfg.training.get("validation_penetration")
    penetration_cfg = (
        {"enabled": True}
        if penetration_cfg is None
        else OmegaConf.to_container(penetration_cfg, resolve=True)
    )
    bounds = cfg.training.get("validation_region_bounds")
    bounds = DEFAULT_REGION_BOUNDS if bounds is None else [float(v) for v in bounds]
    shared = shared_mesh(dataset)
    first = dataset.graphs[0]
    facets_present = all(
        getattr(first, name, None) is not None
        for name in ("contact_faces", "contact_surface_exclusions", "shell_thickness")
    )
    logger0.info(
        f"Evaluating epoch {epoch} of {cfg.training.ckpt_path} on {dist.world_size} "
        f"rank(s): {len(dataset)} held-out cases, regions {region_names(bounds)}; "
        f"records under {output_dir}"
    )
    with torch.no_grad():
        result = run_suite(
            model,
            dataset,
            names,
            data_stats,
            rollout_steps,
            dist.device,
            num_nodes=int(first.num_nodes),
            world_size=dist.world_size,
            rank=dist.rank,
            amp=amp,
            amp_dtype=amp_dtype,
            penetration_cfg=penetration_cfg,
            region_bounds=bounds,
            reference_x=reference_mm(dataset)[:, 0] if shared else None,
            node_errors=bool(cfg.training.get("validation_node_errors", True)),
            facets_present=facets_present,
            logger=logger0,
        )
    if (
        result.probe is not None
        and getattr(first, "rigid_wall", None) is None
        and dict(penetration_cfg).get("wall_position_mm") is None
    ):
        result.notes["wall"] = (
            "no rigid wall (datapipe.rigid_wall_position unset): wall "
            "penetration statistics skipped"
        )
    # The datapipe's frame interval (its default of 5 ms when the recipe sets
    # model.dt to null); the time labels of the record follow it.
    dt_s = float(getattr(dataset, "dt", 5e-3))
    record = dict(
        penetration=result.penetration,
        regions=result.regions,
        node_error_percentiles=result.node_percentiles,
        notes=result.notes,
        cases=result.cases,
        kept_nodes=result.kept_nodes,
        mesh_shared=shared,
        region_bounds_mm=[float(b) for b in bounds],
        dt_s=dt_s,
        metric=(
            "normalized position MSE pooled over every held-out "
            "(case, step, node, channel) entry"
        ),
        **metric_fields(result, data_stats["node"]["pos_std"], dt_s),
    )
    record["epoch"] = int(epoch)
    record["checkpoint"] = str(cfg.training.ckpt_path)
    record["config"] = OmegaConf.to_container(cfg.model, resolve=True)
    log_summary(logger0, epoch, result)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(output_dir / f"epoch_{epoch:04d}.json", record)
        if result.node_rmse is not None:
            save_node_rmse(
                output_dir / f"node_rmse_epoch_{epoch:04d}.npy", result.node_rmse
            )
        logger0.info(f"Record written to {output_dir / f'epoch_{epoch:04d}.json'}")


if __name__ == "__main__":
    main()
