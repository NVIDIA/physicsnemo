# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Train a latent-conditioned FGN weather model."""

import hydra
from omegaconf import DictConfig
from utils.trainer import Trainer

from physicsnemo.distributed import DistributedManager


def run_training(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    trainer = Trainer(cfg)
    trainer.train()


@hydra.main(version_base=None, config_path="config", config_name="fgn")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


if __name__ == "__main__":
    main()
