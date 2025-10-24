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
import numpy as np
import torch
from hydra.utils import to_absolute_path
from physicsnemo.datapipes.gnn.stokes_dataset_dgl import StokesDataset
from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.utils import load_checkpoint
from physicsnemo.models.meshgraphnet import MeshGraphNet
from omegaconf import DictConfig

from utils import relative_lp_error

try:
    from dgl import DGLGraph
    from dgl.dataloading import GraphDataLoader
except:
    raise ImportError(
        "Stokes  example requires the DGL library. Install the "
        + "desired CUDA version at: \n https://www.dgl.ai/pages/start.html"
    )

try:
    import pyvista as pv
except:
    raise ImportError(
        "Stokes  Dataset requires the pyvista library. Install with "
        + "pip install pyvista"
    )


class MGNRollout:
    def __init__(self, cfg: DictConfig, logger):
        self.logger = logger
        self.results_dir = cfg.results_dir

        # set device
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using {self.device} device")

        # instantiate dataset
        self.dataset = StokesDataset(
            name="stokes_test",
            data_dir=to_absolute_path(cfg.data_dir),
            split="test",
            num_samples=cfg.num_test_samples,
        )

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            self.dataset,
            batch_size=cfg.batch_size,
            shuffle=False,
            drop_last=False,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            cfg.input_dim_nodes,
            cfg.input_dim_edges,
            cfg.output_dim,
            aggregation=cfg.aggregation,
            hidden_dim_node_encoder=256,
            hidden_dim_edge_encoder=256,
            hidden_dim_node_decoder=256,
        )
        self.model = self.model.to(self.device)

        # enable train mode
        self.model.eval()

        # load checkpoint
        _ = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=self.model,
            device=self.device,
        )

    def predict(self):
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

        self.pred, self.exact, self.faces, self.graphs = [], [], [], []
        stats = {
            key: value.to(self.device) for key, value in self.dataset.node_stats.items()
        }
        for i, graph in enumerate(self.dataloader):
            graph = graph.to(self.device)
            pred = self.model(graph.ndata["x"], graph.edata["x"], graph).detach()

            keys = ["u", "v", "p"]
            polydata = pv.read(self.dataset.data_list[i])

            for key_index, key in enumerate(keys):
                pred_val = pred[:, key_index : key_index + 1]
                target_val = graph.ndata["y"][:, key_index : key_index + 1]

                pred_val = self.dataset.denormalize(
                    pred_val, stats[f"{key}_mean"], stats[f"{key}_std"]
                )
                target_val = self.dataset.denormalize(
                    target_val, stats[f"{key}_mean"], stats[f"{key}_std"]
                )

                error = relative_lp_error(pred_val, target_val)
                self.logger.info(f"Sample {i} - l2 error of {key}(%): {error:.3f}")

                polydata[f"pred_{key}"] = pred_val.detach().cpu().numpy()

            self.logger.info("-" * 50)
            os.makedirs(to_absolute_path(self.results_dir), exist_ok=True)
            polydata.save(
                os.path.join(to_absolute_path(self.results_dir), f"graph_{i}.vtp")
            )


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    logger = PythonLogger("main")  # General python logger
    logger.file_logging()

    logger.info("Rollout started...")
    rollout = MGNRollout(cfg, logger)
    rollout.predict()


if __name__ == "__main__":
    main()
