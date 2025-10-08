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

import logging
from pathlib import Path

from dgl import DGLGraph
from dgl.dataloading import GraphDataLoader

import hydra
from hydra.utils import instantiate, to_absolute_path

import numpy as np
import pyvista as pv
import torch

from omegaconf import DictConfig

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.launch.utils import load_checkpoint

from loggers import init_python_logging
from utils import batch_as_dict


logger = logging.getLogger("agnet")


def dgl_to_pyvista(graph: DGLGraph):
    """
    Converts a DGL graph to a PyVista graph.

    Parameters:
    -----------
    graph: DGLGraph
        The input DGL graph.

    Returns:
    --------
    pv_graph:
        The output PyVista graph.
    """

    pv_graph = pv.PolyData()

    # Assuming "pos" is in the source graph node data.
    assert "pos" in graph.ndata, f"pos data does not exist, {graph.ndata.keys()=}"
    pv_graph.points = graph.ndata["pos"].numpy()

    # Create lines from edges.
    edges = np.column_stack(graph.edges())
    lines = np.empty((edges.shape[0], 3), dtype=np.int64)
    lines[:, 0] = 2
    lines[:, 1:] = edges

    pv_graph.lines = lines.flatten()
    pv_graph.point_data["p_pred"] = graph.ndata["p_pred"].numpy()
    pv_graph.point_data["p"] = graph.ndata["p"].numpy()
    pv_graph.point_data["wallShearStress_pred"] = graph.ndata[
        "wallShearStress_pred"
    ].numpy()
    pv_graph.point_data["wallShearStress"] = graph.ndata["wallShearStress"].numpy()

    return pv_graph


class EvalRollout:
    """MGN inference with a given experiment."""

    def __init__(self, cfg: DictConfig):
        self.output_dir = Path(to_absolute_path(cfg.output))
        logger.info(f"Storing results in {self.output_dir}")

        self.device = DistributedManager().device
        logger.info(f"Using {self.device} device")

        # instantiate dataset
        logger.info("Loading the test dataset...")
        self.dataset = instantiate(cfg.data.test)
        logger.info(f"Using {len(self.dataset)} test samples.")

        # instantiate dataloader
        logger.info("Creating the dataloader...")
        self.dataloader = GraphDataLoader(
            self.dataset,
            **cfg.test.dataloader,
        )

        # instantiate the model
        logger.info("Creating the model...")
        self.model = instantiate(cfg.model).to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        load_checkpoint(
            to_absolute_path(cfg.resume_dir),
            models=self.model,
            device=self.device,
        )

        # instantiate losses.
        logger.info("Creating the losses...")
        self.loss = instantiate(cfg.loss)

    @torch.inference_mode()
    def predict(self, save_results=False):
        """
        Run the prediction process.

        Parameters:
        -----------
        save_results: bool
            Whether to save the results in form of a .vtp file, by default False

        Returns:
        --------
        None
        """

        for batch in self.dataloader:
            graph, case_id, normals, areas, coeff = batch
            assert len(case_id) == 1, "Only batch size 1 is currently supported."

            case_id = case_id[0].item()
            graph = graph.to(self.device)
            normals = normals.to(self.device)[0]
            areas = areas.to(self.device)[0]
            coeff = coeff.to(self.device)[0]

            logger.info(f"Processing case id {case_id}")
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph)
            gt = graph.ndata["y"]
            pred, gt = self.dataset.denormalize(pred, gt, pred.device)

            num_out_c = gt.shape[1]
            if num_out_c in [1, 4]:
                graph.ndata["p_pred"] = pred[:, 0]
                graph.ndata["p"] = gt[:, 0]
            if num_out_c in [3, 4]:
                graph.ndata["wallShearStress_pred"] = pred[:, num_out_c - 3 :]
                graph.ndata["wallShearStress"] = gt[:, num_out_c - 3 :]

            error = self.loss.graph(pred, gt)
            logger.info(f"Error (%): {error * 100:.4f}")

            if save_results:
                # Convert DGL graph to PyVista graph and save it
                pv_graph = dgl_to_pyvista(graph.cpu())
                pv_graph.save(self.output_dir / f"{case_id}.vtp")


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # initialize distributed manager
    DistributedManager.initialize()

    init_python_logging(cfg, DistributedManager().rank)

    logger.info("Rollout started...")
    rollout = EvalRollout(cfg)
    rollout.predict(save_results=True)


if __name__ == "__main__":
    main()
