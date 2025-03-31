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

from collections.abc import Mapping

import matplotlib as mpl

mpl.use("Agg")  # Set the backend to Agg for headless environments
import os
import unittest
import warnings

from hydra.core.hydra_config import HydraConfig

from jaxtyping import Float, Int
from torch import Tensor

try:
    import wandb
except ImportError:
    warnings.warn("wandb is not installed. wandb logger is not available.")

from pathlib import Path
from typing import Any, Dict, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from physicsnemo.distributed import DistributedManager

from src.utils import rank0

from src.utils.visualization import fig_to_numpy


class Logger:
    """Base logger class."""

    def __init__(self):
        pass

    def log_scalar(self, tag: str, value: float, step: int):
        raise NotImplementedError

    def log_image(self, tag: str, img: torch.Tensor, step: int):
        raise NotImplementedError

    def log_time(self, tag: str, duration: float, step: int):
        raise NotImplementedError

    def log_figure(self, tag: str, fig: mpl.figure.Figure, step: int):
        fig.set_tight_layout(True)
        # set the background to white
        fig.patch.set_facecolor("white")
        im = fig_to_numpy(fig)
        self.log_image(tag, im, step)

    def log_dict(self, dict: Dict, step: int):
        for k, v in dict.items():
            self.log_scalar(k, v, step)

    def log_pointcloud(self, tag: str, vertices: Tensor, colors: Tensor, step: int):
        raise NotImplementedError


class TensorBoardLogger(Logger):
    """TensorBoard logger."""

    def __init__(self, log_dir: str):
        super().__init__()
        self.writer = SummaryWriter(log_dir)

    def log_scalar(self, tag: str, value: float, step: int):
        self.writer.add_scalar(tag, value, step)

    def log_image(
        self, tag: str, img: Union[Float[Tensor, "H W C"], np.ndarray], step: int
    ):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        # if img has three axes, assert the last channel has size 3
        if img.ndim == 3:
            assert img.shape[-1] == 3
        self.writer.add_image(tag, img, step, dataformats="HWC")

    def log_time(self, tag: str, duration: float, step: int):
        self.writer.add_scalar(tag, duration, step)

    def log_pointcloud(
        self,
        tag: str,
        vertices: Float[Tensor, "B N 3"],
        colors: Int[Tensor, "B N 3"],
        step: int,
    ):
        # When there is no batch size, add a batch dimension
        if vertices.ndim == 2:
            vertices = vertices.unsqueeze(0)
        if colors.ndim == 2:
            colors = colors.unsqueeze(0)
        # No edges.
        # self.writer.add_mesh(tag, vertices, colors, step)


class WandBLogger(Logger):
    """Weights & Biases logger."""

    def __init__(
        self,
        project_name: str,
        run_name: str,
        log_dir: str,
        group: Optional[str] = None,
        config: Optional[Dict] = None,
        entity: Optional[str] = None,
        resume: Optional[bool] = False,
        wandb_id: Optional[str] = None,
        mode: Optional[str] = "online",
    ):
        super().__init__()
        if resume:
            resume = "must"
            assert wandb_id is not None, "id must be provided when resuming"
        else:
            resume = "allow"

        if wandb_id is None:
            wandb_id = wandb.util.generate_id()
            # Save wandb_id if it exists in config and has not been written
            wandb_id_file = os.path.join(config.output, "wandb_id.txt")
            if not os.path.exists(wandb_id_file):
                with open(os.path.join(config.output, "wandb_id.txt"), "w") as f:
                    f.write(wandb_id)

        wandb.init(
            project=project_name,
            name=run_name,
            group=group,
            dir=log_dir,
            entity=entity,
            resume=resume,
            id=wandb_id,
            mode=mode,
        )
        # log config to wandb
        if config is not None and resume != "must":
            wandb.config.update(flatten_dict(config, sep="."), allow_val_change=True)

        # Save config.
        wandb.save(
            os.path.join(config.output, HydraConfig.get().output_subdir, "config.yaml"),
            base_path=config.output,
            policy="now",
        )

    def log_scalar(self, tag: str, value: float, step: int):
        wandb.log({tag: value}, step=step)

    def log_image(
        self, tag: str, img: Union[Float[Tensor, "H W C"], np.ndarray], step: int
    ):
        if isinstance(img, torch.Tensor):
            img = img.numpy()
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        # if img has three axes, assert the last channel has size 3
        if img.ndim == 3:
            assert img.shape[-1] == 3
        wandb.log({tag: [wandb.Image(img)]}, step=step)

    def log_time(self, tag: str, duration: float, step: int):
        wandb.log({tag: duration}, step=step)

    def log_pointcloud(
        self,
        tag: str,
        vertices: Float[Tensor, "B N 3"],
        colors: Int[Tensor, "B N 3"],
        step: int,
    ):
        if vertices.ndim == 2:
            vertices = vertices.unsqueeze(0)
        if colors.ndim == 2:
            colors = colors.unsqueeze(0)

        assert vertices.shape[0] == colors.shape[0]
        assert vertices.shape[1] == colors.shape[1]
        assert vertices.shape[2] == 3
        assert colors.shape[2] == 3

        for i in range(vertices.shape[0]):
            vertices_colors = torch.cat((vertices[i], colors[i].float()), dim=-1)
            wandb.log(
                {f"{tag}_{i}": wandb.Object3D(vertices_colors.cpu().numpy())}, step=step
            )


class Loggers(Logger):
    """A class that wraps multiple loggers."""

    def __init__(self, loggers: Union[Logger, list]):
        super().__init__()

        if isinstance(loggers, list):
            self.loggers = loggers
        else:
            self.loggers = [loggers]

    @rank0
    def log_scalar(self, tag: str, value: float, step: int):
        for logger in self.loggers:
            logger.log_scalar(tag, value, step)

    @rank0
    def log_image(self, tag: str, img: torch.Tensor, step: int):
        for logger in self.loggers:
            logger.log_image(tag, img, step)

    @rank0
    def log_figure(self, tag: str, fig, step: int):
        for logger in self.loggers:
            logger.log_figure(tag, fig, step)

    @rank0
    def log_time(self, tag: str, duration: float, step: int):
        for logger in self.loggers:
            logger.log_time(tag, duration, step)

    @rank0
    def log_dict(self, dict: Dict, step: int):
        for logger in self.loggers:
            logger.log_dict(dict, step)

    @rank0
    def log_pointcloud(
        self,
        tag: str,
        vertices: Float[Tensor, "B N 3"],
        colors: Int[Tensor, "B N 3"],
        step: int,
    ):
        for logger in self.loggers:
            logger.log_pointcloud(tag, vertices, colors, step)


def init_logger(config: dict) -> Logger:

    if DistributedManager().rank != 0:
        return Loggers([])

    loggers = []

    for logger_type, logger_cfg in config.loggers.items():
        if logger_type == "tensorboard":
            loggers.append(TensorBoardLogger(log_dir=config.log_dir))
        elif logger_type == "wandb":
            # Check if the path exists and has at least one checkpoint
            resume = False
            wandb_id = None
            output_dir = config.output
            checkpoints = []
            if output_dir is not None:
                output_dir = Path(output_dir)
                if output_dir.exists():
                    checkpoints = list(output_dir.glob("*.pth"))

            if len(checkpoints) > 0:
                resume = True
                # Find the wandb_id text file
                wandb_id_file = output_dir / "wandb_id.txt"
                assert (
                    wandb_id_file.exists()
                ), f"wandb_id.txt not found in output dir: {output_dir}"
                with open(wandb_id_file) as f:
                    wandb_id = f.read()
                print(f"Resuming wandb run: {wandb_id}")

            loggers.append(
                WandBLogger(
                    logger_cfg.project_name,
                    logger_cfg.run_name,
                    entity=logger_cfg.entity,
                    group=logger_cfg.group_name,
                    log_dir=config.log_dir,
                    config=config,
                    resume=resume,
                    wandb_id=wandb_id,
                    mode=logger_cfg.mode,
                )
            )
        else:
            raise ValueError(f"Unknown logger type: {logger_type}")

    return Loggers(loggers)


def flatten_dict(
    d: Mapping[str, Any],
    parent_key: str = "",
    sep: str = "_",
    no_sep_keys: tuple[str] = ("base",),
) -> dict[str, Any]:
    items = []
    for k, v in d.items():
        # Do not expand parent key if it is "base"
        if parent_key in no_sep_keys:
            new_key = k
        else:
            new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, Mapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class TestLoggers(unittest.TestCase):
    """Loggers unit tests class."""

    def setUp(self) -> None:
        # Generate some example data
        X = np.arange(-5, 5, 0.25)
        Y = np.arange(-5, 5, 0.25)
        X, Y = np.meshgrid(X, Y)
        R = np.sqrt(X**2 + Y**2)
        Z = np.sin(R)
        self.test_data = (X, Y, Z)

        return super().setUp()

    def test_tensorboard(self):
        logger = TensorBoardLogger("test")
        logger.log_scalar("test", 1.0, 0)
        logger.log_image("test", torch.rand(64, 64, 3), 0)
        logger.log_time("test", 1.0, 0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(*self.test_data, cmap="viridis")
        logger.log_figure("test", fig, 0)

    def test_wandb(self):
        logger = WandBLogger("test", "test", "test")
        logger.log_scalar("test", 1.0, 0)
        logger.log_image("test", torch.rand(64, 64, 3), 0)
        logger.log_time("test", 1.0, 0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_surface(*self.test_data, cmap="viridis")
        logger.log_figure("test", fig, 0)

    def test_loggers(self):
        loggers = Loggers(
            [TensorBoardLogger("test"), WandBLogger("test", "test", "test")]
        )
        loggers.log_scalar("test", 1.0, 0)
        loggers.log_image("test", torch.rand(64, 64, 3), 0)
        loggers.log_time("test", 1.0, 0)


if __name__ == "__main__":
    unittest.main()
