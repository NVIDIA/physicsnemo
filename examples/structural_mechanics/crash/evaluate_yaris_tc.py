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

Runs the validation of :class:`yaris_full_trainer.YarisFullTrainer` on a
trained checkpoint, inside the training environment and sharded over GPUs
like training: pooled and per-frame vector RMSE in millimetres, per-case and
per-region breakdowns, the target-relative penetration statistics of the
prediction and the ground truth (shell surface and, with
``datapipe.rigid_wall_position``, the rigid wall), optionally node-error maps
and the kinematics readout::

    torchrun --standalone --nproc_per_node=4 evaluate_yaris_tc.py \\
        --config-name yaris_deformer \\
        yaris.data_root=... yaris.split_manifest=... yaris.outlier_mask=... \\
        yaris.work_dir=... training.ckpt_path=<run>/checkpoints \\
        +training.validation_output_dir=<out> hydra.run.dir=<out> \\
        datapipe.contact_surface=true [+training.load_epoch=30] \\
        [+training.validation_node_errors=true +training.validation_kinematics=true]

The run's own validation records are untouched: the JSON goes to
``<out>/epoch_<E>.json`` where ``E`` is the loaded checkpoint's epoch.
``datapipe.contact_surface=true`` gives contact-free models the shell facets the
penetration metrics need; such models ignore them. For the corrector recipe pass
``model.teacher_weights=<teacher .mdlus>`` as in training.
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig, open_dict
from train import resolve_trainer_class

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


@hydra.main(version_base="1.3", config_path="conf", config_name="yaris_deformer")
def main(cfg: DictConfig) -> None:
    """Evaluate the checkpoint of ``training.ckpt_path`` and write the record."""
    DistributedManager.initialize()
    dist = DistributedManager()
    logger0 = RankZeroLoggingWrapper(PythonLogger("evaluate"), dist)
    output_dir = cfg.training.get("validation_output_dir")
    if not output_dir:
        raise ValueError(
            "pass +training.validation_output_dir=<dir>; the run's own validation "
            "directory is never written by this script"
        )
    with open_dict(cfg):
        if "validation_penetration" not in cfg.training:
            cfg.training.validation_penetration = {"enabled": True}
        else:
            cfg.training.validation_penetration.enabled = True
        cfg.training.tensorboard_log_dir = str(Path(output_dir) / "tensorboard")
        # Weights only: the run's RNG checkpoint belongs to its own world size.
        cfg.training.restore_rng = False
    trainer = resolve_trainer_class(cfg.training.get("trainer_class"))(cfg, logger0)
    if trainer.epoch_init <= 0 and not cfg.training.get("warm_start_model"):
        raise RuntimeError(f"No checkpoint found under {cfg.training.ckpt_path}")
    logger0.info(
        f"Evaluating epoch {trainer.epoch_init} of {cfg.training.ckpt_path} "
        f"on {dist.world_size} rank(s); records under {output_dir}"
    )
    # A warm-started evaluation (external weights) has no training checkpoint;
    # its record is written as epoch 0.
    stats = trainer.validate(max(trainer.epoch_init - 1, -1))
    logger0.info(
        f"Validation epoch {trainer.epoch_init}: MSE: {stats['MSE'].item():.6f}"
    )


if __name__ == "__main__":
    main()
