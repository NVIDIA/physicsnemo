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

"""Train the PhysicsNeMo FNO local solver used by FP-DDM."""

from pathlib import Path

import hydra
import torch
from fpddm.training import train_model
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import LaunchLogger, PythonLogger


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Train FP-DDM from the shared Hydra configuration."""

    DistributedManager.initialize()
    dist = DistributedManager()
    torch.set_float32_matmul_precision("medium")
    logger = PythonLogger("fp_ddm_train")
    LaunchLogger.initialize()

    cfg.training.output_dir = to_absolute_path(cfg.training.output_dir)
    if cfg.training.resume_dir:
        cfg.training.resume_dir = to_absolute_path(cfg.training.resume_dir)
    output_dir = Path(cfg.training.output_dir)
    if dist.rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, output_dir / "config.yaml")

    config = OmegaConf.to_container(cfg, resolve=True)
    best_checkpoint = train_model(config, dist)
    if dist.rank == 0:
        logger.success(f"Best checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
