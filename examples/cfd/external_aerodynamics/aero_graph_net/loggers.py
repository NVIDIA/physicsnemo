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


from abc import ABC, abstractmethod
import functools
import logging

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import wandb

from physicsnemo.distributed import DistributedManager


def init_python_logging(config: DictConfig, rank: int = 0) -> None:
    """Initializes Python logging."""

    pylog_cfg = OmegaConf.select(config, "logging.python")
    if pylog_cfg is None:
        return

    # Set up Python loggers.
    pylog_cfg.output = config.output
    pylog_cfg.rank = rank
    # Enable logging only on rank 0, if requested.
    if pylog_cfg.rank0_only and pylog_cfg.rank != 0:
        pylog_cfg.handlers = {}
        pylog_cfg.loggers.agnet.handlers = []
    # Configure logging.
    logging.config.dictConfig(OmegaConf.to_container(pylog_cfg, resolve=True))


def rank0(func):
    """Decorator that allows the function to be executed only in rank 0 process."""

    @functools.wraps(func)
    def rank0_only(*args, **kwargs):
        if DistributedManager().rank == 0:
            func(*args, **kwargs)

    return rank0_only


class ExperimentLogger(ABC):
    """Provides unified interface to a logger."""

    @abstractmethod
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        pass

    @abstractmethod
    def log_image(self, tag: str, value, step: int) -> None:
        pass


class WandBLogger(ExperimentLogger):
    """Wrapper for Weights & Biases logger."""

    def __init__(self, **kwargs) -> None:
        if DistributedManager().rank != 0:
            return
        wandb.init(**kwargs)

    @rank0
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        wandb.log({tag: value}, step=step)

    @rank0
    def log_image(self, tag: str, value, step: int) -> None:
        wandb.log({tag: wandb.Image(value)}, step=step)


class CompositeLogger(ExperimentLogger):
    """Wraps a list of loggers providing unified interface."""

    loggers: dict[str, ExperimentLogger] = None

    def __init__(self, config: DictConfig) -> None:
        if DistributedManager().rank != 0:
            self.loggers = {}
            return
        # Instantiate loggers only when running on rank 0.
        self.loggers = instantiate(config.loggers)

    @rank0
    def log_scalar(self, tag: str, value: float, step: int) -> None:
        for logger in self.loggers.values():
            logger.log_scalar(tag, value, step)

    @rank0
    def log_image(self, tag: str, value: float, step: int) -> None:
        for logger in self.loggers.values():
            logger.log_image(tag, value, step)
