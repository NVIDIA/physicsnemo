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

import os

import hydra
from hydra.utils import to_absolute_path

from dgl.dataloading import GraphDataLoader
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import tri as mtri
from matplotlib.patches import Rectangle
import numpy as np
from omegaconf import DictConfig
import torch

from physicsnemo.models.meshgraphnet import MeshGraphNet
from physicsnemo.datapipes.gnn.deforming_plate_dataset import DeformingPlateDataset
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.utils import load_checkpoint

import numpy as np


def extract_surface_triangles(tets):
    # tets: (N_tet, 4) array of indices
    # Returns: (N_surface_tri, 3) array of triangle indices
    faces = np.concatenate(
        [
            tets[:, [0, 1, 2]],
            tets[:, [0, 1, 3]],
            tets[:, [0, 2, 3]],
            tets[:, [1, 2, 3]],
        ],
        axis=0,
    )
    # Sort each face so that duplicates can be found
    faces = np.sort(faces, axis=1)
    # Find unique faces and their counts
    faces_tuple = [tuple(face) for face in faces]
    from collections import Counter

    face_counts = Counter(faces_tuple)
    # Surface faces appear only once
    surface_faces = np.array(
        [face for face, count in face_counts.items() if count == 1]
    )
    return surface_faces


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger: PythonLogger):
        self.num_test_time_steps = cfg.num_test_time_steps
        self.frame_skip = cfg.frame_skip

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = DeformingPlateDataset(
            name="deforming_plate_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
            num_steps=cfg.num_test_time_steps,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn="silu" if cfg.recompute_activation else "relu",
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

        self.var_identifier = {"ux": 0, "uy": 1, "uz": 2, "stress": 3, "disp_mag": -1}

    def predict(self):
        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }
        for i, (graph, cells, mask) in enumerate(self.dataloader):
            graph = graph.to(self.device)
            # denormalize data
            graph.ndata["x"][:, 0:3] = self.dataset.denormalize(
                graph.ndata["x"][:, 0:3],
                stats["world_pos_mean"],
                stats["world_pos_std"],
            )
            graph.ndata["y"][:, 0:3] = self.dataset.denormalize(
                graph.ndata["y"][:, 0:3],
                stats["world_pos_diff_mean"],
                stats["world_pos_diff_std"],
            )
            graph.ndata["y"][:, [3]] = self.dataset.denormalize(
                graph.ndata["y"][:, [3]],
                stats["stress_mean"],
                stats["stress_std"],
            )

            # inference step
            invar = graph.ndata["x"].clone()

            if i % (self.num_test_time_steps - 1) != 0:
                invar[:, 0:3] = self.pred[i - 1][:, 0:3].clone()
                i += 1
            invar[:, 0:3] = self.dataset.normalize_node(
                invar[:, 0:3], stats["world_pos_mean"], stats["world_pos_std"]
            )
            pred_i = self.model(invar, graph.edata["x"], graph).detach()  # predict

            # denormalize prediction
            pred_i[:, 0:3] = self.dataset.denormalize(
                pred_i[:, 0:3],
                stats["world_pos_diff_mean"],
                stats["world_pos_diff_std"],
            )
            pred_i[:, 3] = self.dataset.denormalize(
                pred_i[:, 3], stats["stress_mean"], stats["stress_std"]
            )
            invar[:, 0:3] = self.dataset.denormalize(
                invar[:, 0:3], stats["world_pos_mean"], stats["world_pos_std"]
            )

            # do not update the "wall_boundary" & "outflow" nodes
            mask = torch.cat((mask, mask, mask), dim=-1).to(self.device)
            pred_i[:, 0:3] = torch.where(
                mask, pred_i[:, 0:3], torch.zeros_like(pred_i[:, 0:3])
            )

            # integration
            self.pred.append(
                torch.cat(
                    ((pred_i[:, 0:3] + invar[:, 0:3]), pred_i[:, [3]]), dim=-1
                ).cpu()
            )
            self.exact.append(
                torch.cat(
                    (
                        (graph.ndata["y"][:, 0:3] + graph.ndata["x"][:, 0:3]),
                        graph.ndata["y"][:, [3]],
                    ),
                    dim=-1,
                ).cpu()
            )

            self.faces.append(torch.squeeze(cells).numpy())
            self.graphs.append(graph.cpu())

    def get_raw_data(self, idx):
        # Support for displacement magnitude
        if idx == -1:  # -1 will be used for disp_mag
            self.pred_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.pred]
            self.exact_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.exact]
        else:
            self.pred_i = [var[:, idx] for var in self.pred]
            self.exact_i = [var[:, idx] for var in self.exact]
        return self.graphs, self.faces, self.pred_i, self.exact_i

    def init_animation(self, idx):
        # Support for displacement magnitude
        if idx == -1:  # -1 will be used for disp_mag
            self.pred_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.pred]
            self.exact_i = [torch.linalg.norm(var[:, 0:3], dim=1) for var in self.exact]
        else:
            self.pred_i = [var[:, idx] for var in self.pred]
            self.exact_i = [var[:, idx] for var in self.exact]

        # fig configs
        plt.rcParams["image.cmap"] = "inferno"
        self.fig, self.ax = plt.subplots(1, 2, figsize=(16, 9))

        # Set background color to black
        self.fig.set_facecolor("black")
        self.ax[0].set_facecolor("black")
        self.ax[1].set_facecolor("black")

        # make animations dir
        if not os.path.exists("./animations"):
            os.makedirs("./animations")

    def animate(self, num):
        num *= self.frame_skip
        graph = self.graphs[num]
        y_star = self.pred_i[num].numpy()
        y_exact = self.exact_i[num].numpy()
        cells = self.faces[num]
        surface_tris = extract_surface_triangles(cells)

        # For predicted mesh
        mesh_pos_pred = (graph.ndata["x"][:, 0:3] + self.pred[num][:, 0:3]).numpy()
        stress_pred = self.pred[num][:, 3].numpy()

        # For ground truth mesh
        mesh_pos_exact = (graph.ndata["x"][:, 0:3] + self.exact[num][:, 0:3]).numpy()
        stress_exact = self.exact[num][:, 3].numpy()

        disp_pred = np.linalg.norm(self.pred[num][:, 0:3].numpy(), axis=1)
        disp_exact = np.linalg.norm(self.exact[num][:, 0:3].numpy(), axis=1)

        # Now plot using PolyCollection or trisurf (for 3D)
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        self.ax[0].cla()
        self.ax[0] = self.fig.add_subplot(1, 2, 1, projection="3d")
        tris = mesh_pos_pred[surface_tris]
        # Use a solid metallic color (e.g., 'silver')
        col = Poly3DCollection(tris, facecolor="silver", edgecolor="k", linewidths=0.05)
        self.ax[0].add_collection3d(col)
        self.ax[0].auto_scale_xyz(
            mesh_pos_pred[:, 0], mesh_pos_pred[:, 1], mesh_pos_pred[:, 2]
        )
        self.ax[0].set_title("Predicted Deformed Mesh", color="white")

        self.ax[1].cla()
        self.ax[1] = self.fig.add_subplot(1, 2, 2, projection="3d")
        tris = mesh_pos_exact[surface_tris]
        col = Poly3DCollection(tris, facecolor="silver", edgecolor="k", linewidths=0.05)
        self.ax[1].add_collection3d(col)
        self.ax[1].auto_scale_xyz(
            mesh_pos_exact[:, 0], mesh_pos_exact[:, 1], mesh_pos_exact[:, 2]
        )
        self.ax[1].set_title("True Deformed Mesh", color="white")

        # Adjust subplots to minimize empty space
        self.ax[0].set_aspect("auto", adjustable="box")
        self.ax[1].set_aspect("auto", adjustable="box")
        self.ax[0].autoscale(enable=True, tight=True)
        self.ax[1].autoscale(enable=True, tight=True)
        self.fig.subplots_adjust(
            left=0.05, bottom=0.05, right=0.95, top=0.95, wspace=0.2, hspace=0.05
        )
        return self.fig


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()
    logger.info("Rollout started...")
    rollout = MGNRollout(cfg, logger)
    idx = [rollout.var_identifier[k] for k in cfg.viz_vars]
    rollout.predict()

    for k, i in zip(cfg.viz_vars, idx):
        rollout.init_animation(i)
        ani = animation.FuncAnimation(
            rollout.fig,
            rollout.animate,
            frames=len(rollout.graphs) // cfg.frame_skip,
            interval=cfg.frame_interval,
        )
        ani.save(f"animations/animation_{k}.gif")
        logger.info(f"Created animation for {k}")


if __name__ == "__main__":
    main()
