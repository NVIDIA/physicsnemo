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

"""Run the FP-DDM thermal domain-decomposition workflow."""

from pathlib import Path

import hydra
from fpddm.pipeline import run_fpddm
from fpddm.training import set_seed
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils.logging import PythonLogger


@hydra.main(version_base="1.3", config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    """Run FP-DDM from the shared Hydra configuration."""

    DistributedManager.initialize()
    dist = DistributedManager()
    if dist.world_size != 1:
        raise RuntimeError("FP-DDM Schwarz inference currently supports one process")
    logger = PythonLogger("fp_ddm")
    set_seed(int(cfg.run.seed))

    cfg.run.output_dir = to_absolute_path(cfg.run.output_dir)
    if cfg.run.checkpoint_dir:
        cfg.run.checkpoint_dir = to_absolute_path(cfg.run.checkpoint_dir)
    output_dir = Path(cfg.run.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, output_dir / "config.yaml")

    result = run_fpddm(OmegaConf.to_container(cfg, resolve=True), device=dist.device)
    logger.success(
        f"Completed {len(result.metrics)} iterations in "
        f"{result.elapsed_seconds:.2f} seconds"
    )


if __name__ == "__main__":
    main()
