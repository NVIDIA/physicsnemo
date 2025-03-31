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

from functools import partial
import logging
import os
from typing import Any

import hydra
from hydra.utils import instantiate, to_absolute_path

import dgl
from dgl.dataloading import GraphDataLoader

import matplotlib
from matplotlib import animation
from matplotlib import pyplot as plt

matplotlib.use("TkAgg")  # for plotting

import numpy as np

from omegaconf import DictConfig, OmegaConf

import torch
from torch import Tensor

from physicsnemo.datapipes.gnn.lagrangian_dataset import graph_update
from physicsnemo.launch.utils import load_checkpoint

from loggers import get_gpu_info, init_python_logging


logger = logging.getLogger("lmgn")


# From DeepMind's code in render_rollout.py
TYPE_TO_COLOR = {
    0: "green",  # Rigid solids.
    3: "black",  # Boundary particles.
    5: "blue",  # Water.
    6: "gold",  # Sand.
    7: "magenta",  # Goop.
}


class MGNRollout:
    def __init__(self, cfg: DictConfig):

        if cfg.test.batch_size != 1:
            raise ValueError(
                f"Only batch size 1 is currently supported, got {cfg.test.batch_size}"
            )

        self.dim = cfg.dim
        self.frame_skip = cfg.inference.frame_skip
        self.num_history = cfg.data.test.num_history
        self.num_node_type = cfg.data.test.num_node_types
        self.plotting_index = 0

        # set device
        self.device = cfg.test.device
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        logger.info("Loading the test dataset...")
        self.dataset = instantiate(cfg.data.test)
        logger.info(f"Using {len(self.dataset)} test samples.")

        self.num_steps = self.dataset.num_steps
        self.dim = self.dataset.dim
        self.radius = self.dataset.radius
        self.dt = self.dataset.dt
        self.bounds = self.dataset.bounds

        self.time_integrator = self.dataset.time_integrator
        self.compute_boundary_feature = self.dataset.compute_boundary_feature
        self.boundary_clamp = self.dataset.boundary_clamp

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.test.dataloader,
        )

        # instantiate the model
        logger.info("Creating the model...")
        # instantiate the model
        self.model = instantiate(cfg.model)

        if cfg.compile.enabled:
            self.model = torch.compile(self.model, **cfg.compile.args)
        self.model = self.model.to(self.device)

        # enable eval mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model,
            device=self.device,
        )

    @torch.inference_mode()
    def predict(self) -> tuple[Tensor, Tensor, Tensor]:
        pred_pos = []
        gt_pos = []
        node_type = []

        for graph in self.dataloader:
            graph = graph.to(self.device)
            # t == 0 at the start of a new sequence.
            if graph.ndata["t"][0].item() == 0:
                if pred_pos:
                    yield torch.stack(pred_pos), torch.stack(gt_pos), node_type

                # Set initial position, history and node types.
                pred_pos = []
                gt_pos = []
                node_type = []
                position, vel_history, node_type = self.dataset.unpack_inputs(graph)

                pred_pos.append(position)
                gt_pos.append(position)

            graph.ndata["x"] = self.dataset.pack_inputs(
                position, vel_history, node_type
            )
            graph.ndata["pos"] = position
            graph_update(graph, self.radius)

            acceleration = self.model(
                graph.ndata["x"], graph.edata["x"], graph
            )  # predict

            # update the inputs using the prediction from previous iteration
            position, velocity = self.time_integrator(
                position=position,
                velocity=vel_history[-1],
                acceleration=acceleration,
                dt=self.dt,
            )
            position = self.boundary_clamp(position, bounds=self.bounds)
            velocity = self.dataset.normalize_velocity(velocity)
            # Drop the oldest velocity and append the most recent one.
            vel_history = torch.cat((vel_history[1:], velocity.unsqueeze(0)), dim=0)

            pred_pos.append(position)
            gt_pos.append(self.dataset.unpack_targets(graph)[0])

        # Last sequence.
        yield torch.stack(pred_pos), torch.stack(gt_pos), node_type


def init_animation(subplot_kw: dict[str, Any] = None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), subplot_kw=subplot_kw)
    return fig, ax1, ax2


def plot_particles_2d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.set_aspect("equal")
    ax.scatter(position[:, 0], position[:, 1], c=node_color)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_title(title, color="black")


def plot_particles_3d(ax, title, position, node_color, bounds):
    ax.cla()
    ax.set_aspect("equal")
    # ZXY to match axis order in the dataset.
    ax.scatter(position[:, 2], position[:, 0], position[:, 1], c=node_color)
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[0], bounds[1])
    ax.set_zlim(bounds[0], bounds[1])
    ax.set_title(title, color="black")


def animate(num, plotter, fig, ax1, ax2, pred, gt, node_color, bounds, frame_skip):
    num *= frame_skip
    plotter(ax1, "PhysicsNeMo MeshGraphNet Prediction", pred[num], node_color, bounds)
    plotter(ax2, "Ground Truth", gt[num], node_color, bounds)

    fig.subplots_adjust(
        left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.1, hspace=0.2
    )


def plot_error(mse, out_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(mse) + 1))
    for i, (err, color) in enumerate(zip(mse, colors)):
        ax.plot(err, marker=".", linestyle="-", color=color, label=f"{i}", alpha=0.6)
        ax.axhline(err.mean(), linestyle="--", color=color)
    # Global mean.
    m = np.array(mse).mean()
    ax.axhline(m, linestyle="--", color=colors[-1], label="All")
    ax.text(-0.1, m, f"{m:.3f}", color=colors[-1], verticalalignment="bottom")

    ax.set_title("Lagrangian MeshGraphNet")
    ax.set_xlabel("time steps")
    ax.set_ylabel("Position MSE error")
    ax.grid(True)
    ax.legend()

    fig.savefig(os.path.join(out_dir, "error.png"))
    plt.close(fig)


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    init_python_logging(cfg, base_filename="inference")
    logger.info(f"Config summary:\n{OmegaConf.to_yaml(cfg, sort_keys=True)}")
    logger.info(get_gpu_info())

    logger.info("Rollout started...")
    rollout = MGNRollout(cfg)

    ani_dir = os.path.join(cfg.output, "animations")
    os.makedirs(ani_dir, exist_ok=True)

    mse = []
    # test on dataset
    for i, (pred_pos, gt_pos, node_type) in enumerate(rollout.predict()):
        logger.info(f"Processing sequence {i}...")

        pred = pred_pos.cpu().numpy()
        gt = gt_pos.cpu().numpy()
        node_type = node_type.cpu().numpy()
        node_color = [TYPE_TO_COLOR[idx] for idx in np.argmax(node_type, axis=1)]

        # plot
        if cfg.dim == 2:
            fig, ax1, ax2 = init_animation()
            plotter = plot_particles_2d
        elif cfg.dim == 3:
            fig, ax1, ax2 = init_animation(subplot_kw={"projection": "3d"})
            plotter = plot_particles_3d
        else:
            assert False, f"{cfg.dim=}"

        ani_func = partial(
            animate,
            plotter=plotter,
            fig=fig,
            ax1=ax1,
            ax2=ax2,
            pred=pred,
            gt=gt,
            node_color=node_color,
            bounds=rollout.bounds,
            frame_skip=rollout.frame_skip,
        )

        ani = animation.FuncAnimation(
            fig,
            ani_func,
            frames=(rollout.num_steps - rollout.num_history - 1) // rollout.frame_skip,
            interval=cfg.inference.frame_interval,
        )
        ani.save(os.path.join(ani_dir, f"animation_{i}.gif"))
        plt.close(fig)
        logger.info(f"Created animation_{i}.gif")

        # Rollout MSE.
        mse.append(np.mean((pred - gt) ** 2, axis=(1, 2)))

    # Create error plot.
    plot_error(mse, ani_dir)


if __name__ == "__main__":
    main()
