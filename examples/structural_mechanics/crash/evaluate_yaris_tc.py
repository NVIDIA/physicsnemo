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
"""Score a Yaris time-conditional checkpoint with the exact held-out validation
and the crossing-aware penetration statistics of predictions and ground truth.

Runs inside the training environment, sharded over GPUs like training::

    torchrun --standalone --nproc_per_node=4 evaluate_yaris_tc.py \\
        --config-name crash_deformer_knn_yaris_tc_loss_physical \\
        yaris.data_root=... yaris.split_manifest=... yaris.outlier_mask=... \\
        yaris.work_dir=... training.ckpt_path=<run>/checkpoints \\
        +training.validation_output_dir=<out> hydra.run.dir=<out> \\
        datapipe.contact_surface=true [+training.load_epoch=30]

A prediction ensemble averages several weight files of the same
architecture: ``+training.ensemble_models=[a.mdlus,b.mdlus,c.mdlus]`` (the
run's own checkpoint directory may then be empty).

The run's own validation records are untouched: the JSON goes to
``<out>/epoch_<E>.json`` where ``E`` is the loaded checkpoint's epoch.
``datapipe.contact_surface=true`` gives contact-free models the shell facets the
probe needs; such models ignore them.
"""

import copy
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, open_dict
from torch import nn
from train import resolve_trainer_class
from utils import load_warm_start

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper


class PredictionEnsemble(nn.Module):
    """Average the predictions of several copies of one model architecture.

    ``weights`` are ``.mdlus`` (or state-dict ``.pt``) files; each is loaded
    into a deep copy of ``template`` with :func:`load_warm_start` (strict on
    unexpected tensors). Evaluation only: outputs must be tensors.
    """

    def __init__(self, template: nn.Module, weights: list[str]):
        super().__init__()
        template = getattr(template, "module", template)
        members = []
        for path in weights:
            member = copy.deepcopy(template)
            load_warm_start(member, path, allowed_missing_prefixes=())
            members.append(member.eval())
        self.members = nn.ModuleList(members)

    def forward(self, *args, **kwargs):
        outputs = [member(*args, **kwargs) for member in self.members]
        if not all(isinstance(out, torch.Tensor) for out in outputs):
            raise TypeError("PredictionEnsemble averages tensor outputs only")
        return torch.stack(outputs).mean(dim=0)


@hydra.main(
    version_base="1.3",
    config_path="conf",
    config_name="crash_deformer_knn_yaris_tc_loss_physical",
)
def main(cfg: DictConfig) -> None:
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
    members = cfg.training.get("ensemble_models")
    if members:
        # Prediction ensemble: every listed weight file becomes a copy of the
        # configured model; their outputs are averaged in evaluation mode.
        trainer.model = PredictionEnsemble(trainer.model, list(members)).to(dist.device)
        logger0.info(f"Prediction ensemble of {len(members)} models")
    elif trainer.epoch_init <= 0 and not cfg.training.get("warm_start_model"):
        raise RuntimeError(f"No checkpoint found under {cfg.training.ckpt_path}")
    logger0.info(
        f"Evaluating epoch {trainer.epoch_init} of {cfg.training.ckpt_path} "
        f"on {dist.world_size} rank(s); records under {output_dir}"
    )
    # A warm-started evaluation (checkpoint soup, external weights) has no
    # training checkpoint; its record is written as epoch 0.
    stats = trainer.validate(max(trainer.epoch_init - 1, -1))
    logger0.info(
        f"Validation epoch {trainer.epoch_init}: MSE: {stats['MSE'].item():.6f}"
    )


if __name__ == "__main__":
    main()
