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

"""Train diffusion-based generative model using the techniques described in the
paper "Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import warnings

import hydra
from omegaconf import DictConfig
import torch

from physicsnemo.distributed import DistributedManager

from utils.trainer import Trainer


@hydra.main(version_base=None, config_path="config", config_name="regression")
def main(cfg: DictConfig) -> None:
    """Train regression or diffusion models for use in the StormCast (https://arxiv.org/abs/2408.10958) ML-based weather model"""

    # suppress nuisance warnings
    warnings.filterwarnings(
        "ignore", message="^.*``NO_SHARD`` for ``ShardingStrategy``.*$"
    )
    warnings.filterwarnings("ignore", message="^.*`_get_pg_default_device.*$")
    warnings.filterwarnings(
        "ignore", message="^.*`NO_SHARD` sharding strategy is deprecated.*$"
    )
    warnings.filterwarnings(
        "ignore", message="^.*You are importing from 'physicsnemo.experimental'.*$"
    )

    # Initialize
    DistributedManager.initialize()

    # Random seed.
    if cfg.training.seed < 0:
        seed = torch.randint(1 << 31, size=[], device=torch.device("cuda"))
        torch.distributed.broadcast(seed, src=0)
        cfg.training.seed = int(seed)

    # Start from specified checkpoint, if provided
    if cfg.training.initial_weights is not None:
        weights_path = cfg.training.initial_weights
        if not os.path.isfile(weights_path) or not (
            weights_path.endswith(".mdlus") or weights_path.endswith(".pt")
        ):
            raise ValueError(
                "training.initial_weights must point to a physicsnemo .mdlus or .pt checkpoint from a previous training run"
            )

    # Set up rundir if not existing
    os.makedirs(cfg.training.rundir, exist_ok=True)

    # Train.
    trainer = Trainer(cfg)
    trainer.train()


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
