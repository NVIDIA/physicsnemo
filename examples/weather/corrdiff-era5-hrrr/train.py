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

"""Train an ERA5 -> HRRR downscaling model with x-prediction rectified flow
matching, following "Back to Basics: Let Denoising Generative Models Denoise"
(https://arxiv.org/abs/2511.13720)."""

import os
import warnings

import hydra
from omegaconf import DictConfig

from physicsnemo.distributed import DistributedManager

from utils.trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="diffusion")
def main(cfg: DictConfig) -> None:
    """Train the corrdiff-era5-hrrr downscaling model"""

    # The scheduler warns that the reverse-process drift is singular at t = 1.
    # This example trains up to t = 1, which is safe: the x-prediction loss and
    # the flow used for sampling stay finite at t = 1.
    warnings.filterwarnings(
        "ignore", message="^RectifiedFlowNoiseScheduler was constructed with t_max=1.0"
    )

    # Initialize
    DistributedManager.initialize()

    # Set up rundir if not existing
    os.makedirs(cfg.training.rundir, exist_ok=True)

    # Train.
    trainer = Trainer(cfg)
    trainer.train()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
