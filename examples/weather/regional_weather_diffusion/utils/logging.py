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

"""Logging functions."""

from dataclasses import asdict
import glob
import os

import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import wandb

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper

from utils.config import MainConfig


class ExperimentLogger:
    """Wraps logging functions in one place."""

    def __init__(self, name: str, cfg: MainConfig):
        dist = DistributedManager()
        self.log_to_wandb = cfg.training.log_to_wandb and (dist.rank == 0)
        self.log_to_tensorboard = cfg.training.log_to_tensorboard and (dist.rank == 0)

        self.logger0 = RankZeroLoggingWrapper(PythonLogger(name), dist)

        if self.log_to_wandb:
            wandb_resume = False
            os.makedirs(cfg.training.rundir, exist_ok=True)
            net_name = (
                "regression" if cfg.training.loss.type == "regression" else "diffusion"
            )
            training_states = glob.glob(
                os.path.join(
                    cfg.training.rundir, f"checkpoints_{net_name}/checkpoint*.pt"
                )
            )
            if training_states:
                wandb_resume = True

            entity, project = "wandb_entity", "wandb_project"
            wandb.init(
                dir=cfg.training.rundir,
                config=asdict(cfg),
                name=os.path.basename(cfg.training.rundir),
                project=project,
                entity=entity,
                resume=wandb_resume,
                mode=cfg.training.wandb_mode,
            )
            self.wandb_logs = {}
            self.info("WandB logging enabled")

        self.tensorboard_writer = None
        if self.log_to_tensorboard:
            tb_dir = os.path.join(cfg.training.rundir, "tensorboard")
            os.makedirs(tb_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=tb_dir)
            self.info(f"TensorBoard logging enabled: {tb_dir}")

        self.step = 0

    def info(self, info: str):
        """Print an info string (printed from rank 0 only)."""
        self.logger0.info(info)

    def log_value(self, key: str, value: float):
        """Log a numerical value (e.g. a loss)."""
        if self.log_to_wandb:
            self.wandb_logs[key] = value

        if self.log_to_tensorboard:
            self.tensorboard_writer.add_scalar(key, value, self.step)

    def log_figure(self, key: str, fig: plt.Figure):
        """Log a matplotlib figure."""
        if self.log_to_wandb:
            self.wandb_logs[key] = wandb.Image(fig)

        if self.log_to_tensorboard:
            self.tensorboard_writer.add_figure(key, fig, self.step)

    def dump(self):
        """Write out logged values."""
        if self.log_to_wandb:
            wandb.log(self.wandb_logs, step=self.step)

        if self.log_to_tensorboard:
            self.tensorboard_writer.flush()

    def finalize(self):
        """Close loggers."""
        if self.log_to_wandb:
            wandb.finish()

        if self.log_to_tensorboard:
            self.tensorboard_writer.close()
