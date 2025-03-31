# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import hydra
import torch
import wandb
import glob
from omegaconf import DictConfig, OmegaConf
from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from utils.trainer import training_loop


@hydra.main(version_base=None, config_path="config", config_name="regression")
def main(cfg: DictConfig) -> None:
    """Train regression or diffusion models for use in the StormCast (https://arxiv.org/abs/2408.10958) ML-based weather model"""

    # Initialize
    DistributedManager.initialize()
    dist = DistributedManager()
    logger = PythonLogger("main")
    logger0 = RankZeroLoggingWrapper(logger, dist)  # Log only from rank 0

    # Random seed.
    if cfg.training.seed < 0:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        cfg.training.seed = int(seed)

    # Resume from specified checkpoint, if provided
    if cfg.training.resume_checkpoint is not None:
        resume = cfg.training.resume_checkpoint
        if not os.path.isfile(resume) or not resume.endswith(".pt"):
            raise ValueError(
                "training.resume_checkpoint must point to a physicsnemo .pt checkpoint from a previous training run"
            )

    # If run directory already exists, then resume training from last checkpoint
    wandb_resume = False
    if os.path.exists(cfg.training.rundir):
        training_states = sorted(
            glob.glob(os.path.join(cfg.training.rundir, "checkpoints/checkpoint*.pt"))
        )
        if training_states:
            logger0.info(
                "Resuming training from previous run_dir: " + cfg.training.rundir
            )
            last_training_state = training_states[-1]
            cfg.training.resume_checkpoint = last_training_state
            logger0.info(
                "Resuming training from previous checkpoint file: "
                + last_training_state
            )
            wandb_resume = True

    # Setup wandb, if enabled
    if dist.rank == 0 and cfg.training.log_to_wandb:
        entity, project = "wandb_entity", "wandb_project"
        wandb.init(
            dir=cfg.training.rundir,
            config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
            name=os.path.basename(cfg.training.rundir),
            project=project,
            entity=entity,
            resume=wandb_resume,
            mode="online",
        )

    # Train.
    training_loop(cfg)


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
