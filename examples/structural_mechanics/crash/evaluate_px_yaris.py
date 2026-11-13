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
"""Score a Yaris checkpoint with the exact held-out validation suite.

Runs the suite of :mod:`metrics.validation_suite` (the one
:class:`trainers.px_yaris_trainer.YarisFullTrainer` validates with) on a trained
checkpoint, inside the training environment and sharded over GPUs like
training: pooled and per-frame vector RMSE in millimetres, per-case and
per-region breakdowns, the target-relative penetration statistics of the
prediction and the ground truth (shell surface and, with
``datapipe.rigid_wall_position``, the rigid wall) and optionally node-error
maps::

    torchrun --standalone --nproc_per_node=4 evaluate_px_yaris.py \\
        --config-name px_yaris_deformer \\
        yaris.data_root=... yaris.split_manifest=... yaris.outlier_mask=... \\
        yaris.work_dir=<run> training.ckpt_path=<run>/checkpoints \\
        training.validation_output_dir=<out> hydra.run.dir=<out>/hydra \\
        datapipe.contact_surface=true [training.load_epoch=30] \\
        [training.evaluation_split=test] [training.validation_node_errors=true]

Only the requested split (``training.evaluation_split``: ``validation`` by
default, or ``test``) is built, in load mode: the frozen statistics, the kNN
statistics and the neighbourhood cache are read from the run's work directory
(``datapipe.stats_dir``, ``${yaris.work_dir}/stats`` by default) and never
recomputed, so a missing or foreign work directory fails instead of silently
rebuilding statistics; no training dataset, optimizer or TensorBoard writer is
created. ``hydra.run.dir`` must lie inside ``training.validation_output_dir``
(the recipe's run directory is the training run, whose ``.hydra`` files would
otherwise be overwritten). The JSON goes to ``<out>/epoch_<E>.json`` where
``E`` is the loaded checkpoint's epoch (0 for a weights-only file given as
``training.warm_start_model``). ``datapipe.contact_surface=true`` gives
contact-free models the shell facets the penetration metrics need; such models
ignore them. For the corrector recipe pass ``model.teacher_weights=<teacher
.mdlus>`` as in training. The penetration metrics are this script's purpose, so
it exits with an error (after writing the record, whose penetration block then
holds ``null`` values and the reasons) when the probe was disabled on any rank.
"""

from pathlib import Path

import hydra
import torch
from hydra.utils import instantiate
from metrics.validation_suite import (
    amp_settings,
    data_stats_of,
    load_evaluation_weights,
    log_summary,
    require_output_run_dir,
)
from omegaconf import DictConfig, OmegaConf, open_dict
from trainers.px_yaris_trainer import (
    SPLITS,
    build_split_dataset,
    evaluate_split,
    write_outputs,
)

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


@hydra.main(version_base="1.3", config_path="conf", config_name="px_yaris_deformer")
def main(cfg: DictConfig) -> None:
    """Score the checkpoint of ``training.ckpt_path`` on one split and write the record."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(PythonLogger("evaluate"), dist)
    output_dir = require_output_run_dir(cfg.training.get("validation_output_dir"))
    split = str(cfg.training.get("evaluation_split", "validation"))
    if split not in SPLITS:
        raise ValueError(
            f"training.evaluation_split must be one of {SPLITS}; got {split!r}"
        )
    with open_dict(cfg):
        # The penetration metrics are the script's purpose.
        cfg.training.validation_penetration.enabled = True
    if "teacher_weights" in cfg.model and not OmegaConf.is_missing(
        cfg.model, "teacher_weights"
    ):
        # A corrector recipe names the frozen network's checkpoint; a wrong path
        # fails here, before the dataset is read.
        teacher_weights = str(cfg.model.teacher_weights)
        if not Path(teacher_weights).is_file():
            raise FileNotFoundError(
                f"model.teacher_weights={teacher_weights} does not exist; it must "
                "be the trained network's .mdlus file"
            )
    stats_dir = Path(str(cfg.datapipe.stats_dir))
    if not stats_dir.is_dir():
        raise FileNotFoundError(
            f"datapipe.stats_dir={stats_dir} does not exist; point it at the "
            "training run's statistics directory (nothing is recomputed here)"
        )

    reader = instantiate(cfg.reader)
    dataset = build_split_dataset(cfg, reader, split, logger0)
    data_stats = data_stats_of(dataset, dist.device)
    model = instantiate(cfg.model)
    epoch = load_evaluation_weights(
        model, cfg, dist.device, logger0, world_size=dist.world_size
    )
    model.to(dist.device).eval()
    amp, amp_dtype = amp_settings(cfg)
    logger0.info(
        f"Evaluating epoch {epoch} of {cfg.training.ckpt_path} on the {split} split "
        f"({len(dataset)} cases, {dataset.num_nodes} kept nodes) over "
        f"{dist.world_size} rank(s); records under {output_dir}"
    )
    with torch.no_grad():
        result = evaluate_split(
            model,
            dataset,
            data_stats,
            cfg,
            dist.device,
            world_size=dist.world_size,
            rank=dist.rank,
            amp=amp,
            amp_dtype=amp_dtype,
            logger=logger0,
        )
    log_summary(logger0, epoch, result)
    if dist.rank == 0:
        write_outputs(result, dataset, data_stats, epoch, output_dir)
        logger0.info(f"Record written to {output_dir / f'epoch_{epoch:04d}.json'}")
    penetration = result.penetration
    if penetration is None or not penetration["complete"]:
        # The record is written (with null penetration values and the reasons);
        # the numbers this script exists for are missing, so fail loudly.
        reasons = "; ".join(penetration["failure_reasons"]) if penetration else "none"
        raise RuntimeError(
            "Penetration metrics were disabled during the evaluation "
            f"(reasons: {reasons}); the record under {output_dir} carries null "
            "penetration values"
        )


if __name__ == "__main__":
    main()
