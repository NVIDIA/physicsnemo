# ignore_header_test
# Copyright 2023 Stanford University
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

import torch
from dgl.dataloading import GraphDataLoader
from torch.cuda.amp import GradScaler
import time, os
import numpy as np
import hydra

from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.models.meshgraphnet import MeshGraphNet

# from physicsnemo.datapipes.gnn.mgn_dataset import MGNDataset
import generate_dataset as gd
from generate_dataset import generate_normalized_graphs
from generate_dataset import train_test_split
from generate_dataset import Bloodflow1DDataset

from physicsnemo.launch.logging import (
    PythonLogger,
    RankZeroLoggingWrapper,
)
from physicsnemo.launch.logging.wandb import initialize_wandb
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint
import json
from omegaconf import DictConfig


def mse(input, target, mask):
    """
    Mean square error.

    This is defined as the ((input - target)**2).mean()

    Arguments:
        input: first tensor
        target: second tensor (ideally, the result we are trying to match)
        mask: tensor of weights for loss entries with same size as input and
              target.

    Returns:
        The mean square error

    """
    return (mask * (input - target) ** 2).mean()


class MGNTrainer:
    def __init__(self, logger, cfg, dist):
        # set device
        self.device = dist.device
        logger.info(f"Using {self.device} device")

        norm_type = {"features": "normal", "labels": "normal"}
        graphs, params = generate_normalized_graphs(
            "raw_dataset/graphs/", norm_type, cfg.training.geometries
        )

        graph = graphs[list(graphs)[0]]

        infeat_nodes = graph.ndata["nfeatures"].shape[1] + 1
        infeat_edges = graph.edata["efeatures"].shape[1]
        nout = 2

        nodes_features = [
            "area",
            "tangent",
            "type",
            "T",
            "dip",
            "sysp",
            "resistance1",
            "capacitance",
            "resistance2",
            "loading",
        ]

        edges_features = ["rel_position", "distance", "type"]

        params["infeat_nodes"] = infeat_nodes
        params["infeat_edges"] = infeat_edges
        params["out_size"] = nout
        params["node_features"] = nodes_features
        params["edges_features"] = edges_features
        params["rate_noise"] = cfg.training.rate_noise
        params["stride"] = cfg.training.stride

        trainset, testset = train_test_split(graphs, cfg.training.train_test_split)

        train_graphs = [graphs[gname] for gname in trainset]
        traindataset = Bloodflow1DDataset(train_graphs, params, trainset)

        # instantiate dataloader
        self.dataloader = GraphDataLoader(
            traindataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
        )

        # instantiate the model
        self.model = MeshGraphNet(
            params["infeat_nodes"],
            params["infeat_edges"],
            2,
            processor_size=cfg.architecture.processor_size,
            hidden_dim_node_encoder=cfg.architecture.hidden_dim_node_encoder,
            hidden_dim_edge_encoder=cfg.architecture.hidden_dim_edge_encoder,
            hidden_dim_processor=cfg.architecture.hidden_dim_processor,
            hidden_dim_node_decoder=cfg.architecture.hidden_dim_node_decoder,
        )

        if cfg.performance.jit:
            self.model = torch.jit.script(self.model).to(self.device)
        else:
            self.model = self.model.to(self.device)

        # enable train mode
        self.model.train()

        # instantiate loss, optimizer, and scheduler
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.scheduler.lr)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=cfg.training.epochs,
            eta_min=cfg.scheduler.lr * cfg.scheduler.lr_decay,
        )
        self.scaler = GradScaler()

        # load checkpoint
        self.epoch_init = load_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.device,
        )

        self.params = params
        self.cfg = cfg

    def backward(self, loss):
        """
        Perform backward pass.

        Arguments:
            loss: loss value.

        """
        # backward pass
        if self.cfg.performance.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

    def train(self, graph):
        """
        Perform one training iteration over one graph. The training is performed
        over multiple timesteps, where the number of timesteps is specified in
        the 'stride' parameter.

        Arguments:
            graph: the desired graph.

        Returns:
            loss: loss value.

        """
        graph = graph.to(self.device)
        self.optimizer.zero_grad()
        loss = 0
        ns = graph.ndata["next_steps"]

        # create mask to weight boundary nodes more in loss
        mask = torch.ones(ns[:, :, 0].shape, device=self.device)
        imask = graph.ndata["inlet_mask"].bool()
        outmask = graph.ndata["outlet_mask"].bool()

        bcoeff = self.cfg.training.loss_weight_boundary_nodes
        mask[imask, 0] = mask[imask, 0] * bcoeff
        # flow rate is known
        mask[outmask, 0] = mask[outmask, 0] * bcoeff
        mask[outmask, 1] = mask[outmask, 1] * bcoeff

        states = [graph.ndata["nfeatures"].clone()]

        nnodes = mask.shape[0]
        nf = torch.zeros((nnodes, 1), device=self.device)
        for istride in range(self.params["stride"]):
            # impose boundary condition
            nf[imask, 0] = ns[imask, 1, istride]
            nfeatures = torch.cat((states[-1], nf), 1)
            pred = self.model(nfeatures, graph.edata["efeatures"], graph)

            # add prediction by MeshGraphNet to current state
            new_state = torch.clone(states[-1])
            new_state[:, 0:2] += pred

            # impose exact flow rate at the inlet (to remove it from loss)
            new_state[imask, 1] = ns[imask, 1, istride]
            states.append(new_state)

            if istride == 0:
                coeff = self.cfg.training.loss_weight_1st_timestep
            else:
                coeff = self.cfg.training.loss_weight_other_timesteps

            loss += coeff * mse(states[-1][:, 0:2], ns[:, :, istride], mask)

        self.backward(loss)

        return loss


@hydra.main(version_base=None, config_path=".", config_name="config")
def do_training(cfg: DictConfig):
    """
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # initialize loggers
    logger = PythonLogger("main")
    logger.file_logging()

    # initialize trainer
    trainer = MGNTrainer(logger, cfg, dist)

    # training loop
    start = time.time()
    logger.info("Training started...")
    for epoch in range(trainer.epoch_init, cfg.training.epochs):
        for graph in trainer.dataloader:
            loss = trainer.train(graph)

        logger.info(
            f"epoch: {epoch}, loss: {loss:10.3e}, time per epoch: {(time.time() - start):10.3e}"
        )

        # save checkpoint
        save_checkpoint(
            os.path.join(cfg.checkpoints.ckpt_path, cfg.checkpoints.ckpt_name),
            models=trainer.model,
            optimizer=trainer.optimizer,
            scheduler=trainer.scheduler,
            scaler=trainer.scaler,
            epoch=epoch,
        )
        start = time.time()
        trainer.scheduler.step()

        with open(cfg.checkpoints.ckpt_path + "/parameters.json", "w") as outf:
            json.dump(trainer.params, outf, indent=4)
    logger.info("Training completed!")


"""
    Perform training over all graphs in the dataset.

    Arguments:
        cfg: Dictionary of parameters.

    """
if __name__ == "__main__":
    do_training()
