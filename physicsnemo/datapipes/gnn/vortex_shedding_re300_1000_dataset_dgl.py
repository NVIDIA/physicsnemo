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

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset
from dgl.dataloading import GraphDataLoader
from tqdm import tqdm

from .utils import load_json, save_json


class LatentDataset(DGLDataset):
    """In-memory Mesh-Reduced-Transformer Dataset in the latent space.
    Notes:
        - Set produce_latents = True when first use this dataset.

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    produce_latents : bool, optional
        Specifying whether to use the trained Encoder to compress the graph into latent space and save the restuls, by default True
    Encoder: torch.nn.Module, optioanl
        The trained model used for encoding, by default None
    position_mesh: torch.Tensor, optioanl
        The postions for all meshes, by default None
    position_pivotal: torch.Tensor, optioanl
        The postions for all pivotal positions+
        , by default None
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self,
        name="dataset",
        data_dir="dataset",
        split="train",
        sequence_len=401,
        produce_latents=True,
        Encoder=None,
        position_mesh=None,
        position_pivotal=None,
        dist=None,
        verbose=False,
    ):
        super().__init__(
            name=name,
            verbose=verbose,
        )
        self.split = split
        self.sequence_len = sequence_len
        self.data_dir = data_dir
        if produce_latents:
            self.save_latents(Encoder, position_mesh, position_pivotal, dist)

        self.z = torch.load("{}/latent_{}.pt".format(self.data_dir, self.split)).cpu()
        self.get_re_number()

    def __len__(self):
        return len(self.z) // self.sequence_len

    def __getitem__(self, idx):
        return (
            self.z[idx * self.sequence_len : (idx + 1) * self.sequence_len],
            self.re[idx : (idx + 1)],
        )

    def get_re_number(self):
        """Get RE number"""
        ReAll = torch.from_numpy(np.linspace(300, 1000, 101)).float().reshape([-1, 1])
        nuAll = 1 / ReAll
        listCatALL = []
        for i in range(3):
            re = ReAll ** (i + 1)
            nu = nuAll ** (i + 1)
            listCatALL.append(re / re.max())
            listCatALL.append(nu / nu.max())
        if self.split == "train":
            index = [i for i in range(101) if i % 2 == 0]
        else:
            index = [i for i in range(101) if i % 2 == 1]

        self.re = torch.cat(listCatALL, dim=1)[index, :]

    @torch.no_grad()
    def save_latents(self, Encoder, position_mesh, position_pivotal, dist):
        Encoder.eval()
        if self.split == "train":
            dataset = VortexSheddingRe300To1000Dataset(
                name="vortex_shedding_train", split="train"
            )

        else:
            dataset = VortexSheddingRe300To1000Dataset(
                name="vortex_shedding_train", split="test"
            )

        dataloader = GraphDataLoader(
            dataset, batch_size=1, shuffle=False, drop_last=False, pin_memory=True
        )
        record_z = []
        for graph in tqdm(dataloader):
            graph = graph.to(dist.device)
            z = Encoder.encode(
                graph.ndata["x"],
                graph.edata["x"],
                graph,
                position_mesh,
                position_pivotal,
            )
            z = z.reshape(1, -1)
            record_z.append(z)
        record_z = torch.cat(record_z, dim=0)
        torch.save(record_z, "{}/latent_{}.pt".format(self.data_dir, self.split))


class VortexSheddingRe300To1000Dataset(DGLDataset):
    """In-memory Mesh-Reduced-Transformer Dataset for stationary mesh.
    Notes:
        - A single adj matrix is used for each transient simulation.
            Do not use with adaptive mesh or remeshing

    Parameters
    ----------
    name : str, optional
        Name of the dataset, by default "dataset"
    data_dir : _type_, optional
        Specifying the directory that stores the raw data in .TFRecord format., by default None
    split : str, optional
        Dataset split ["train", "eval", "test"], by default "train"
    verbose : bool, optional
        verbose, by default False
    """

    def __init__(
        self, name="dataset", data_dir="dataset", split="train", verbose=False
    ):

        super().__init__(
            name=name,
            verbose=verbose,
        )
        self.data_dir = data_dir

        self.split = split
        self.rawData = np.load(
            os.path.join(self.data_dir, "rawData.npy"), allow_pickle=True
        )

        # select training and testing set
        if self.split == "train":
            self.sequence_ids = [i for i in range(101) if i % 2 == 0]
        if self.split == "test":
            self.sequence_ids = [i for i in range(101) if i % 2 == 1]

        # solution states are velocity and pressure
        self.solution_states = torch.from_numpy(
            self.rawData["x"][self.sequence_ids, :, :, :]
        ).float()

        # edge information
        self.E = torch.from_numpy(self.rawData["edge_attr"]).float()

        # edge connection
        self.A = torch.from_numpy(self.rawData["edge_index"]).type(torch.long)

        # sequence length
        self.sequence_len = self.solution_states.shape[1]
        self.sequence_num = self.solution_states.shape[0]
        self.num_nodes = self.solution_states.shape[2]

        if self.split == "train":
            self.edge_stats = self._get_edge_stats()
        else:
            self.edge_stats = load_json("dataset/edge_stats.json")

        if self.split == "train":
            self.node_stats = self._get_node_stats()
        else:
            self.node_stats = load_json("dataset/node_stats.json")

        # handle the normalization
        for i in range(self.sequence_num):
            for j in range(self.sequence_len):
                self.solution_states[i, j] = self.normalize(
                    self.solution_states[i, j],
                    self.node_stats["node_mean"],
                    self.node_stats["node_std"],
                )
        self.E = self.normalize(
            self.E, self.edge_stats["edge_mean"], self.edge_stats["edge_std"]
        )

    def __len__(self):
        return self.sequence_len * self.sequence_num

    def __getitem__(self, idx):
        sidx = idx // self.sequence_len
        tidx = idx % self.sequence_len

        node_features = self.solution_states[sidx, tidx]
        node_targets = self.solution_states[sidx, tidx]
        graph = dgl.graph((self.A[0], self.A[1]), num_nodes=self.num_nodes)
        graph.ndata["x"] = node_features
        graph.ndata["y"] = node_targets
        graph.edata["x"] = self.E
        return graph

    def _get_edge_stats(self):
        stats = {
            "edge_mean": self.E.mean(dim=0),
            "edge_std": self.E.std(dim=0),
        }
        save_json(stats, "dataset/edge_stats.json")
        return stats

    def _get_node_stats(self):
        stats = {
            "node_mean": self.solution_states.mean(dim=[0, 1, 2]),
            "node_std": self.solution_states.std(dim=[0, 1, 2]),
        }
        save_json(stats, "dataset/node_stats.json")
        return stats

    @staticmethod
    def normalize(invar, mu, std):
        """normalizes a tensor"""
        if invar.size()[-1] != mu.size()[-1] or invar.size()[-1] != std.size()[-1]:
            raise ValueError(
                "invar, mu, and std must have the same size in the last dimension"
            )
        return (invar - mu.expand(invar.size())) / std.expand(invar.size())

    @staticmethod
    def denormalize(invar, mu, std):
        """denormalizes a tensor"""
        denormalized_invar = invar * std + mu
        return denormalized_invar
